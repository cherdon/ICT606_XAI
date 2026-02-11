"""
Hierarchical Binary Classification - Stacking Ensemble (SVM + RF + XGBoost)

Stage 1: Filter out Bad wines (quality < 6) vs Regular wines (quality >= 6)
Stage 2: Among Regular wines, classify Premium (>=7) vs Non-Premium (<7)

================================================================================
STACKING ENSEMBLE STRATEGY
================================================================================

Base Models (Level 0):
- SVM (RBF kernel) - Best individual performer
- Random Forest - Captures non-linear interactions
- XGBoost - Gradient boosting perspective

Meta-Learner (Level 1):
- SVM with RBF kernel (no Logistic Regression as per requirement)

Why Stacking Works:
1. Combines diverse model perspectives
2. Meta-learner learns optimal weighting
3. Reduces variance through averaging
4. Can capture patterns individual models miss

================================================================================
FEATURE ENGINEERING (Wine Industry Domain Knowledge)
================================================================================

1. SO2 Ratio (Stability Index): free SO2 / total SO2
2. Molecular SO2 (Active Preservative): Free SO2 / (1 + 10^(pH - 1.81))
3. VA to FA Ratio (Spoilage Indicator): volatile acidity / fixed acidity
4. Sugar to Acidity (Dryness/"Trocken" Rule): residual sugar / total acidity
5. Alcohol to Density (Structure/Body): alcohol / density

================================================================================
"""
import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, confusion_matrix
)
import xgboost as xgb

from preprocessing import (
    load_wine_data, handle_missing_values, remove_duplicates,
    apply_yeo_johnson_transform
)
from experiments.experiment_tracker import save_experiment_results

EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Stage 1 threshold: Bad wine is quality < 6 (balanced class distribution)
BAD_WINE_THRESHOLD = 6


def engineer_wine_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Create domain-specific wine features based on industry knowledge."""
    df = df.copy()
    
    if verbose:
        print("\n[Feature Engineering] Creating domain-specific wine features...")
    
    df['so2_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-8)
    df['molecular_so2'] = df['free sulfur dioxide'] / (1 + 10 ** (df['pH'] - 1.81))
    df['va_to_fa_ratio'] = df['volatile acidity'] / (df['fixed acidity'] + 1e-8)
    total_acidity = df['fixed acidity'] + df['volatile acidity']
    df['sugar_to_acidity'] = df['residual sugar'] / (total_acidity + 1e-8)
    df['alcohol_to_density'] = df['alcohol'] / df['density']
    
    if verbose:
        print("  Created: so2_ratio, molecular_so2, va_to_fa_ratio, sugar_to_acidity, alcohol_to_density")
        print("\n  Feature correlations with quality:")
        for feat in ['so2_ratio', 'molecular_so2', 'va_to_fa_ratio', 'sugar_to_acidity', 'alcohol_to_density']:
            corr = df[feat].corr(df['quality'])
            print(f"    {feat}: {corr:.4f}")
    
    return df


def prepare_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\nOriginal quality distribution:")
        print(df['quality'].value_counts().sort_index())
    df = handle_missing_values(df, strategy='drop')
    df = remove_duplicates(df)
    if verbose:
        print(f"\nData shape after cleaning: {df.shape}")
    return df


def find_optimal_threshold_f1(y_true: np.ndarray, y_pred_proba: np.ndarray, verbose: bool = True) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    if verbose:
        print(f"    Optimal threshold (F1): {optimal_threshold:.4f}")
    return optimal_threshold


def find_filtering_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              quality: np.ndarray, verbose: bool = True) -> float:
    """Find threshold that maximizes filtering of Bad wines while keeping Premium wines."""
    thresholds = np.linspace(0.3, 0.8, 50)
    
    best_threshold = 0.5
    best_score = -np.inf
    
    premium_mask = quality >= 7
    bad_mask = quality < BAD_WINE_THRESHOLD
    regular_non_premium_mask = (quality >= BAD_WINE_THRESHOLD) & (quality < 7)
    
    if verbose:
        print(f"\n    Test set composition:")
        print(f"      Bad (quality < {BAD_WINE_THRESHOLD}): {bad_mask.sum()}")
        print(f"      Regular Non-Premium: {regular_non_premium_mask.sum()}")
        print(f"      Premium (quality >= 7): {premium_mask.sum()}")
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        bad_filtered = ((y_pred == 0) & bad_mask).sum()
        premium_filtered = ((y_pred == 0) & premium_mask).sum()
        score = bad_filtered - (premium_filtered * 10)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_result = {
                'bad_filtered': bad_filtered,
                'regular_filtered': ((y_pred == 0) & regular_non_premium_mask).sum(),
                'premium_filtered': premium_filtered
            }
    
    if verbose:
        print(f"\n    Best filtering threshold: {best_threshold:.4f}")
        print(f"    Bad wines filtered: {best_result['bad_filtered']} / {bad_mask.sum()}")
        print(f"    Premium wines lost: {best_result['premium_filtered']} / {premium_mask.sum()}")
    
    return best_threshold


def create_base_estimators(random_state: int = 42) -> List[Tuple[str, object]]:
    """
    Create the base estimators for stacking.
    Returns list of (name, estimator) tuples.
    """
    # SVM - our best individual performer
    svm = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=random_state,
        C=10,
        gamma='scale'
    )
    
    # Random Forest - captures non-linear interactions
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=random_state,
        n_jobs=-1
    )
    
    # XGBoost - gradient boosting perspective
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    return [
        ('svm', svm),
        ('rf', rf),
        ('xgb', xgb_clf)
    ]


def train_stacking_stage1(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    quality_test: np.ndarray,
    class_names: list, verbose: bool = True
) -> Dict:
    """Train Stage 1: Bad vs Regular classification with Stacking Ensemble."""
    stage_name = f"STAGE 1: Bad (<{BAD_WINE_THRESHOLD}) vs Regular (>={BAD_WINE_THRESHOLD})"
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"{stage_name}")
        print("=" * 60)
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            print(f"  {class_names[cls]} ({cls}): {count} samples ({pct:.2f}%)")

    # Create base estimators
    base_estimators = create_base_estimators(RANDOM_STATE)
    
    # Meta-learner: SVM (no LogReg as per requirement)
    meta_learner = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        C=1.0
    )
    
    if verbose:
        print(f"\n[Stacking Ensemble Configuration]")
        print(f"  Base models: SVM, Random Forest, XGBoost")
        print(f"  Meta-learner: SVM (RBF kernel)")
        print(f"  CV for stacking: {CV_FOLDS}-fold")
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=CV_FOLDS,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False  # Only use base model predictions
    )
    
    if verbose:
        print(f"\nTraining stacking ensemble...")
    
    stacking_clf.fit(X_train, y_train)
    
    # Calibrate probabilities
    if verbose:
        print(f"Applying probability calibration...")
    calibrated_stack = CalibratedClassifierCV(stacking_clf, method='isotonic', cv=5)
    calibrated_stack.fit(X_train, y_train)
    
    y_pred_proba = calibrated_stack.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    if verbose:
        print(f"\nFinding filtering-optimized threshold:")
    optimal_threshold = find_filtering_threshold(y_test, y_pred_proba, quality_test, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    
    if verbose:
        print(f"\n--- {stage_name} Results ---")
        print(f"Accuracy: {accuracy_optimized:.4f}")
        print(f"F1-Score: {f1_optimized:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:   {recall:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
    
    # Get individual base model performance for comparison
    if verbose:
        print(f"\n  Base Model Individual Performance:")
        for name, model in base_estimators:
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = model.decision_function(X_test)
            base_auc = roc_auc_score(y_test, proba)
            base_pred = model.predict(X_test)
            base_f1 = f1_score(y_test, base_pred)
            print(f"    {name.upper():5s}: F1={base_f1:.4f}, AUC={base_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': calibrated_stack, 'stacking_model': stacking_clf,
        'accuracy': accuracy_optimized, 'f1': f1_optimized,
        'auc_roc': auc_roc, 'precision': precision, 'recall': recall,
        'optimal_threshold': optimal_threshold, 'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm, 'class_names': class_names
    }


def train_stacking_stage2(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    class_names: list, verbose: bool = True
) -> Dict:
    """Train Stage 2: Premium vs Non-Premium with Stacking Ensemble."""
    stage_name = "STAGE 2: Premium (>=7) vs Non-Premium (<7)"
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"{stage_name}")
        print("=" * 60)
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            print(f"  {class_names[cls]} ({cls}): {count} samples ({pct:.2f}%)")

    # Create base estimators with adjusted weights for imbalanced Stage 2
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # SVM with balanced weights
    svm = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        C=10,
        gamma='scale'
    )
    
    # Random Forest with balanced_subsample
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # XGBoost with scale_pos_weight
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    base_estimators = [('svm', svm), ('rf', rf), ('xgb', xgb_clf)]
    
    # Meta-learner: SVM
    meta_learner = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        C=1.0
    )
    
    if verbose:
        print(f"\n[Stacking Ensemble Configuration]")
        print(f"  Base models: SVM, Random Forest, XGBoost")
        print(f"  Meta-learner: SVM (RBF kernel)")
        print(f"  XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=CV_FOLDS,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False
    )
    
    if verbose:
        print(f"\nTraining stacking ensemble...")
    
    stacking_clf.fit(X_train, y_train)
    
    # Calibrate probabilities
    if verbose:
        print(f"Applying probability calibration...")
    calibrated_stack = CalibratedClassifierCV(stacking_clf, method='isotonic', cv=5)
    calibrated_stack.fit(X_train, y_train)
    
    y_pred_proba = calibrated_stack.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    if verbose:
        print(f"\nThreshold optimization:")
    optimal_threshold = find_optimal_threshold_f1(y_test, y_pred_proba, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    
    if verbose:
        print(f"\n--- {stage_name} Results ---")
        print(f"Accuracy: {accuracy_optimized:.4f}")
        print(f"F1-Score: {f1_optimized:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
    
    # Get individual base model performance
    if verbose:
        print(f"\n  Base Model Individual Performance:")
        for name, model in base_estimators:
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = model.decision_function(X_test)
            base_auc = roc_auc_score(y_test, proba)
            base_pred = model.predict(X_test)
            base_f1 = f1_score(y_test, base_pred)
            print(f"    {name.upper():5s}: F1={base_f1:.4f}, AUC={base_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': calibrated_stack, 'stacking_model': stacking_clf,
        'accuracy': accuracy_optimized, 'f1': f1_optimized,
        'auc_roc': auc_roc, 'precision': precision, 'recall': recall,
        'optimal_threshold': optimal_threshold, 'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm, 'class_names': class_names
    }


def predict_hierarchical(X, stage1_model, stage2_model, stage1_threshold, stage2_threshold):
    """Hierarchical prediction: Stage 1 filters, Stage 2 classifies remaining."""
    stage1_proba = stage1_model.predict_proba(X)[:, 1]
    stage1_pred = (stage1_proba >= stage1_threshold).astype(int)
    
    final_pred = np.zeros(len(X), dtype=int)
    final_proba = np.zeros(len(X))
    
    regular_mask = stage1_pred == 1
    if regular_mask.sum() > 0:
        stage2_proba = stage2_model.predict_proba(X[regular_mask])[:, 1]
        stage2_pred = (stage2_proba >= stage2_threshold).astype(int)
        final_pred[regular_mask] = stage2_pred
        final_proba[regular_mask] = stage1_proba[regular_mask] * stage2_proba
    
    return final_pred, final_proba


def evaluate_combined(y_true_premium, y_pred, y_pred_proba, verbose=True):
    auc_roc = roc_auc_score(y_true_premium, y_pred_proba)
    
    if verbose:
        print(f"\nCombined threshold optimization:")
    optimal_threshold = find_optimal_threshold_f1(y_true_premium, y_pred_proba, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimized = accuracy_score(y_true_premium, y_pred_optimized)
    f1_optimized = f1_score(y_true_premium, y_pred_optimized)
    precision_optimized = precision_score(y_true_premium, y_pred_optimized)
    recall_optimized = recall_score(y_true_premium, y_pred_optimized)
    
    cm = confusion_matrix(y_true_premium, y_pred_optimized)
    
    return {
        'accuracy': accuracy_optimized, 'f1': f1_optimized, 'auc_roc': auc_roc,
        'precision': precision_optimized, 'recall': recall_optimized,
        'optimal_threshold': optimal_threshold, 'confusion_matrix': cm,
        'class_names': ['Non-Premium', 'Premium']
    }


def plot_all_confusion_matrices(stage1_results, stage2_results, combined_results, 
                                 output_dir, experiment_name):
    """Plot all three confusion matrices in a single figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Stacking Ensemble - {experiment_name}', fontsize=14, fontweight='bold')
    
    sns.heatmap(stage1_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=stage1_results['class_names'], 
                yticklabels=stage1_results['class_names'], ax=axes[0])
    axes[0].set_title(f"Stage 1: Bad vs Regular\nF1={stage1_results['f1']:.4f}")
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(stage2_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=stage2_results['class_names'],
                yticklabels=stage2_results['class_names'], ax=axes[1])
    axes[1].set_title(f"Stage 2: Premium vs Non-Premium\nF1={stage2_results['f1']:.4f}")
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    sns.heatmap(combined_results['confusion_matrix'], annot=True, fmt='d', cmap='Purples',
                xticklabels=combined_results['class_names'],
                yticklabels=combined_results['class_names'], ax=axes[2])
    axes[2].set_title(f"Combined: Premium vs Non-Premium\nF1={combined_results['f1']:.4f}")
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_name}_confusion_matrices.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrices saved to: {output_path}")
    return output_path


def run_experiment(verbose: bool = True) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    if verbose:
        print("=" * 60)
        print(f"EXPERIMENT: {EXPERIMENT_NAME}")
        print("=" * 60)
        print(f"\nMODEL: Stacking Ensemble (SVM + RF + XGBoost)")
        print(f"META-LEARNER: SVM (no LogReg)")
        print(f"\nSTRATEGY: Prune Bad wines (quality < {BAD_WINE_THRESHOLD})")
        print(f"  Stage 1: Bad (<{BAD_WINE_THRESHOLD}) vs Regular (>={BAD_WINE_THRESHOLD})")
        print(f"  Stage 2: Premium (>=7) vs Non-Premium (<7) among Regular")
        print("\n[Step 1] Loading red wine data...")
    
    df = load_wine_data(data_path, wine_type='red')
    df = prepare_data(df.copy(), verbose=verbose)
    
    # Feature Engineering
    df = engineer_wine_features(df, verbose=verbose)
    
    df['is_premium'] = (df['quality'] >= 7).astype(int)
    
    if verbose:
        print("\n[Note] Multicollinearity filter disabled")
        print(f"       Using all {len(df.columns) - 3} features (including 5 engineered)")
    
    df_clean = df.copy()
    df_clean['is_regular'] = (df_clean['quality'] >= BAD_WINE_THRESHOLD).astype(int)
    df_clean['is_premium'] = (df_clean['quality'] >= 7).astype(int)
    
    target_cols = ['quality', 'is_regular', 'is_premium']
    feature_cols = [col for col in df_clean.columns if col not in target_cols]
    
    if verbose:
        print(f"\nFeature columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
    
    X = df_clean[feature_cols].values
    y_regular = df_clean['is_regular'].values
    y_premium = df_clean['is_premium'].values
    quality = df_clean['quality'].values
    
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=quality, random_state=RANDOM_STATE
    )
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_regular_train, y_regular_test = y_regular[train_idx], y_regular[test_idx]
    y_premium_test = y_premium[test_idx]
    quality_train, quality_test = quality[train_idx], quality[test_idx]
    
    # Apply Yeo-Johnson + RobustScaler (best for SVM in the stack)
    if verbose:
        print("\n[Step 2] Applying Yeo-Johnson transformation...")
    X_train, X_test, _ = apply_yeo_johnson_transform(X_train, X_test, feature_cols, verbose=verbose)
    
    if verbose:
        print("\n[Step 3] Applying RobustScaler...")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Stage 1: Bad vs Regular
    stage1_results = train_stacking_stage1(
        X_train, X_test, y_regular_train, y_regular_test, quality_test,
        [f'Bad (<{BAD_WINE_THRESHOLD})', f'Regular (>={BAD_WINE_THRESHOLD})'], verbose
    )
    
    # Stage 2: Premium vs Non-Premium among Regular wines
    regular_train_mask = quality_train >= BAD_WINE_THRESHOLD
    X_train_regular = X_train[regular_train_mask]
    y_premium_among_regular_train = (quality_train[regular_train_mask] >= 7).astype(int)
    
    regular_test_mask = quality_test >= BAD_WINE_THRESHOLD
    X_test_regular = X_test[regular_test_mask]
    y_premium_among_regular_test = (quality_test[regular_test_mask] >= 7).astype(int)
    
    stage2_results = train_stacking_stage2(
        X_train_regular, X_test_regular, y_premium_among_regular_train, y_premium_among_regular_test,
        ['Non-Premium (<7)', 'Premium (>=7)'], verbose
    )
    
    # Combined prediction
    combined_pred, combined_proba = predict_hierarchical(
        X_test, stage1_results['model'], stage2_results['model'],
        stage1_results['optimal_threshold'], stage2_results['optimal_threshold']
    )
    combined_results = evaluate_combined(y_premium_test, combined_pred, combined_proba, verbose)
    
    # Filtering efficiency
    stage1_pred = (stage1_results['y_pred_proba'] >= stage1_results['optimal_threshold']).astype(int)
    samples_passed_to_stage2 = stage1_pred.sum()
    premium_in_passed = ((stage1_pred == 1) & (quality_test >= 7)).sum()
    bad_filtered = ((stage1_pred == 0) & (quality_test < BAD_WINE_THRESHOLD)).sum()
    total_bad = (quality_test < BAD_WINE_THRESHOLD).sum()
    
    if verbose:
        print(f"\n{'=' * 60}")
        print("FILTERING EFFICIENCY ANALYSIS")
        print("=" * 60)
        print(f"Total test samples: {len(y_regular_test)}")
        print(f"Bad wines filtered: {bad_filtered} / {total_bad} ({bad_filtered/total_bad*100:.1f}%)")
        print(f"Samples passed to Stage 2: {samples_passed_to_stage2}")
        print(f"Premium wines in Stage 2: {premium_in_passed}")
    
    return {
        'experiment_name': EXPERIMENT_NAME, 
        'stage1_results': stage1_results,
        'stage2_results': stage2_results, 
        'combined_results': combined_results,
        'filtering_stats': {
            'total_samples': len(y_regular_test),
            'bad_wines': total_bad,
            'bad_filtered': bad_filtered,
            'passed_to_stage2': samples_passed_to_stage2,
            'premium_in_stage2': premium_in_passed
        },
        'feature_cols': feature_cols,
        'scaler': scaler
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')
    
    experiment_data = run_experiment(verbose=True)
    stage1 = experiment_data['stage1_results']
    stage2 = experiment_data['stage2_results']
    combined = experiment_data['combined_results']
    filtering = experiment_data['filtering_stats']
    
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={'accuracy': combined['accuracy'], 'f1': combined['f1'], 'auc_roc': combined['auc_roc']},
        output_dir=output_dir
    )
    
    plot_all_confusion_matrices(stage1, stage2, combined, output_dir, EXPERIMENT_NAME)
    
    print("\n" + "=" * 60)
    print("STACKING ENSEMBLE CLASSIFICATION COMPLETE")
    print("=" * 60)
    
    print("\n" + "-" * 60)
    print("ENSEMBLE CONFIGURATION")
    print("-" * 60)
    print("  Base Models: SVM, Random Forest, XGBoost")
    print("  Meta-Learner: SVM (RBF kernel)")
    print("  Probability Calibration: Isotonic")
    
    print("\n" + "-" * 60)
    print("FINAL RESULTS SUMMARY")
    print("-" * 60)
    print(f"\n  STAGE 1 (Bad vs Regular):")
    print(f"    F1: {stage1['f1']:.4f}, AUC-ROC: {stage1['auc_roc']:.4f}")
    print(f"\n  STAGE 2 (Premium vs Non-Premium):")
    print(f"    F1: {stage2['f1']:.4f}, AUC-ROC: {stage2['auc_roc']:.4f}")
    print(f"\n  COMBINED:")
    print(f"    Accuracy:  {combined['accuracy']:.4f}")
    print(f"    F1-Score:  {combined['f1']:.4f}")
    print(f"    AUC-ROC:   {combined['auc_roc']:.4f}")
    print(f"    Precision: {combined['precision']:.4f}")
    print(f"    Recall:    {combined['recall']:.4f}")
    
    print("\n  Confusion Matrix:")
    print(pd.DataFrame(combined['confusion_matrix'],
        index=['Non-Premium', 'Premium'],
        columns=['Non-Premium', 'Premium']).to_string().replace('\n', '\n  '))
    
    print(f"\n  Filtering: {filtering['bad_filtered']}/{filtering['bad_wines']} bad wines removed")
    
    return experiment_data


if __name__ == "__main__":
    main()
