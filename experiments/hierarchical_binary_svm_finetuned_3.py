"""
Hierarchical Binary Classification for Wine Quality - SVM with Bad Wine Pruning.

Stage 1: Filter out Bad wines (quality < 5) vs Regular wines (quality >= 5)
Stage 2: Among Regular wines, classify Premium (>=7) vs Non-Premium (<7)

================================================================================
STRATEGY
================================================================================

This approach explicitly prunes out "bad" wines first:
- Stage 1: Bad (<5) vs Regular (>=5) - Remove clearly bad wines
- Stage 2: Premium (>=7) vs Non-Premium (<7) - Standard binary classification

Quality distribution reminder:
- 3: 10 samples   (Bad)
- 4: 53 samples   (Bad)
- 5: 681 samples  (Regular, Non-Premium)
- 6: 638 samples  (Regular, Non-Premium)
- 7: 199 samples  (Regular, Premium)
- 8: 18 samples   (Regular, Premium)

================================================================================
"""
import sys
import os
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, confusion_matrix
)

from preprocessing import (
    load_wine_data, handle_missing_values, remove_duplicates,
    remove_multicollinear_features, apply_yeo_johnson_transform, apply_robust_scaling
)
from experiments.experiment_tracker import save_experiment_results

EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORRELATION_THRESHOLD = 0.7
N_ITER_SEARCH = 25
CV_FOLDS = 5

# Stage 1 threshold: Bad wine is quality < 5
BAD_WINE_THRESHOLD = 5


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
    """
    Find threshold that maximizes filtering of Bad wines while keeping Premium wines.
    """
    thresholds = np.linspace(0.3, 0.8, 50)
    
    best_threshold = 0.5
    best_score = -np.inf
    
    # Identify wine categories
    premium_mask = quality >= 7
    bad_mask = quality < BAD_WINE_THRESHOLD
    regular_non_premium_mask = (quality >= BAD_WINE_THRESHOLD) & (quality < 7)
    
    if verbose:
        print(f"\n    Test set composition:")
        print(f"      Bad (quality < {BAD_WINE_THRESHOLD}): {bad_mask.sum()}")
        print(f"      Regular Non-Premium ({BAD_WINE_THRESHOLD} <= quality < 7): {regular_non_premium_mask.sum()}")
        print(f"      Premium (quality >= 7): {premium_mask.sum()}")
        print(f"\n    Searching for optimal filtering threshold...")
    
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        # Calculate metrics
        bad_filtered = ((y_pred == 0) & bad_mask).sum()
        bad_passed = ((y_pred == 1) & bad_mask).sum()
        
        regular_filtered = ((y_pred == 0) & regular_non_premium_mask).sum()
        regular_passed = ((y_pred == 1) & regular_non_premium_mask).sum()
        
        premium_filtered = ((y_pred == 0) & premium_mask).sum()
        premium_passed = ((y_pred == 1) & premium_mask).sum()
        
        # Score: Maximize Bad filtering, minimize Premium loss
        score = bad_filtered - (premium_filtered * 10)
        
        results.append({
            'threshold': thresh,
            'bad_filtered': bad_filtered,
            'bad_passed': bad_passed,
            'regular_filtered': regular_filtered,
            'premium_filtered': premium_filtered,
            'premium_passed': premium_passed,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    best_result = [r for r in results if r['threshold'] == best_threshold][0]
    
    if verbose:
        print(f"\n    Best filtering threshold: {best_threshold:.4f}")
        print(f"    Bad wines filtered: {best_result['bad_filtered']} / {bad_mask.sum()}")
        print(f"    Regular Non-Premium wines filtered: {best_result['regular_filtered']} / {regular_non_premium_mask.sum()}")
        print(f"    Premium wines lost: {best_result['premium_filtered']} / {premium_mask.sum()}")
    
    return best_threshold


def train_svm_stage1(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    quality_test: np.ndarray,
    class_names: list, verbose: bool = True
) -> Dict:
    """Train Stage 1: Bad vs Regular classification."""
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

    param_distributions = {
        'C': [0.1, 1.0, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced', {0: 1, 1: 1}, {0: 2, 1: 1}]
    }
    
    base_model = SVC(probability=True, random_state=RANDOM_STATE)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=base_model, param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH, cv=cv, scoring='f1', n_jobs=-1,
        verbose=0, random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"\nHyperparameter tuning (optimizing for F1)...")
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    if verbose:
        print(f"Best CV F1: {random_search.best_score_:.4f}")
        print(f"Best params: {random_search.best_params_}")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
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
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': model, 'accuracy': accuracy_optimized, 'f1': f1_optimized,
        'auc_roc': auc_roc, 'precision': precision, 'recall': recall,
        'optimal_threshold': optimal_threshold, 'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm, 'class_names': class_names
    }


def train_svm_stage2(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    class_names: list, verbose: bool = True
) -> Dict:
    """Train Stage 2: Premium vs Non-Premium among Regular wines."""
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

    param_distributions = {
        'C': [0.1, 1.0, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced']
    }
    
    base_model = SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced')
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=base_model, param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH, cv=cv, scoring='f1', n_jobs=-1,
        verbose=0, random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"\nHyperparameter tuning (optimizing for F1)...")
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    if verbose:
        print(f"Best CV F1: {random_search.best_score_:.4f}")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
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
        print(f"Accuracy: {accuracy_optimized:.4f}, F1: {f1_optimized:.4f}, AUC-ROC: {auc_roc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': model, 'accuracy': accuracy_optimized, 'f1': f1_optimized,
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
    
    # Only samples predicted as "Regular" (not Bad) go to Stage 2
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
    fig.suptitle(f'Confusion Matrices - {experiment_name}', fontsize=14, fontweight='bold')
    
    sns.heatmap(stage1_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=stage1_results['class_names'], 
                yticklabels=stage1_results['class_names'], ax=axes[0])
    axes[0].set_title(f"Stage 1: Bad vs Regular\nF1={stage1_results['f1']:.4f}, Acc={stage1_results['accuracy']:.4f}")
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(stage2_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=stage2_results['class_names'],
                yticklabels=stage2_results['class_names'], ax=axes[1])
    axes[1].set_title(f"Stage 2: Premium vs Non-Premium\nF1={stage2_results['f1']:.4f}, Acc={stage2_results['accuracy']:.4f}")
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    sns.heatmap(combined_results['confusion_matrix'], annot=True, fmt='d', cmap='Purples',
                xticklabels=combined_results['class_names'],
                yticklabels=combined_results['class_names'], ax=axes[2])
    axes[2].set_title(f"Combined: Premium vs Non-Premium\nF1={combined_results['f1']:.4f}, Acc={combined_results['accuracy']:.4f}")
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_name}_all_confusion_matrices.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nAll confusion matrices saved to: {output_path}")
    return output_path


def run_experiment(verbose: bool = True) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    if verbose:
        print("=" * 60)
        print(f"EXPERIMENT: {EXPERIMENT_NAME}")
        print("=" * 60)
        print(f"\nSTRATEGY: Prune Bad wines (quality < {BAD_WINE_THRESHOLD})")
        print(f"  Stage 1: Bad (<{BAD_WINE_THRESHOLD}) vs Regular (>={BAD_WINE_THRESHOLD})")
        print(f"  Stage 2: Premium (>=7) vs Non-Premium (<7) among Regular")
        print("\n[Step 1] Loading red wine data...")
    
    df = load_wine_data(data_path, wine_type='red')
    df = prepare_data(df.copy(), verbose=verbose)
    
    df['is_premium'] = (df['quality'] >= 7).astype(int)
    
    df_clean, _ = remove_multicollinear_features(
        df, target_column='is_premium', threshold=CORRELATION_THRESHOLD, verbose=verbose
    )
    
    # Create Stage 1 target: Bad vs Regular
    df_clean['is_regular'] = (df_clean['quality'] >= BAD_WINE_THRESHOLD).astype(int)
    df_clean['is_premium'] = (df_clean['quality'] >= 7).astype(int)
    
    target_cols = ['quality', 'is_regular', 'is_premium']
    feature_cols = [col for col in df_clean.columns if col not in target_cols]
    
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
    
    # Apply transformations for SVM
    if verbose:
        print("\n[Step 2] Applying Yeo-Johnson transformation...")
    X_train, X_test, _ = apply_yeo_johnson_transform(X_train, X_test, feature_cols, verbose=verbose)
    
    if verbose:
        print("\n[Step 3] Applying RobustScaler...")
    X_train, X_test, _ = apply_robust_scaling(X_train, X_test, verbose=verbose)
    
    # Stage 1: Bad vs Regular
    stage1_results = train_svm_stage1(
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
    
    stage2_results = train_svm_stage2(
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
        print(f"Bad wines (quality < {BAD_WINE_THRESHOLD}): {total_bad}")
        print(f"Bad wines filtered: {bad_filtered} / {total_bad} ({bad_filtered/total_bad*100:.1f}%)")
        print(f"Samples passed to Stage 2: {samples_passed_to_stage2}")
        print(f"Premium wines in Stage 2 input: {premium_in_passed}")
        if samples_passed_to_stage2 > 0:
            print(f"Stage 2 Premium ratio: {premium_in_passed/samples_passed_to_stage2*100:.1f}%")
    
    return {
        'experiment_name': EXPERIMENT_NAME, 'stage1_results': stage1_results,
        'stage2_results': stage2_results, 'combined_results': combined_results,
        'filtering_stats': {
            'total_samples': len(y_regular_test),
            'bad_wines': total_bad,
            'bad_filtered': bad_filtered,
            'passed_to_stage2': samples_passed_to_stage2,
            'premium_in_stage2': premium_in_passed
        }
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
    print("HIERARCHICAL CLASSIFICATION (BAD WINE PRUNING) COMPLETE")
    print("=" * 60)
    
    print("\n" + "-" * 60)
    print(f"STAGE 1: Bad (<{BAD_WINE_THRESHOLD}) vs Regular (>={BAD_WINE_THRESHOLD})")
    print("-" * 60)
    print(f"  Accuracy:  {stage1['accuracy']:.4f}")
    print(f"  F1-Score:  {stage1['f1']:.4f}")
    print(f"  AUC-ROC:   {stage1['auc_roc']:.4f}")
    print(f"  Precision: {stage1['precision']:.4f}")
    print(f"  Recall:    {stage1['recall']:.4f}")
    print("\n  Confusion Matrix:")
    print(pd.DataFrame(stage1['confusion_matrix'],
        index=[f'Bad (<{BAD_WINE_THRESHOLD})', f'Regular (>={BAD_WINE_THRESHOLD})'],
        columns=[f'Bad (<{BAD_WINE_THRESHOLD})', f'Regular (>={BAD_WINE_THRESHOLD})']).to_string().replace('\n', '\n  '))
    
    print(f"\n  *** FILTERING RESULTS ***")
    print(f"  Bad wines filtered: {filtering['bad_filtered']} / {filtering['bad_wines']}")
    print(f"  Samples passed to Stage 2: {filtering['passed_to_stage2']} / {filtering['total_samples']}")
    
    print("\n" + "-" * 60)
    print("STAGE 2: Premium (>=7) vs Non-Premium (<7)")
    print("-" * 60)
    print(f"  Accuracy:  {stage2['accuracy']:.4f}")
    print(f"  F1-Score:  {stage2['f1']:.4f}")
    print(f"  AUC-ROC:   {stage2['auc_roc']:.4f}")
    print(f"  Precision: {stage2['precision']:.4f}")
    print(f"  Recall:    {stage2['recall']:.4f}")
    print("\n  Confusion Matrix:")
    print(pd.DataFrame(stage2['confusion_matrix'],
        index=['Non-Premium (<7)', 'Premium (>=7)'],
        columns=['Non-Premium (<7)', 'Premium (>=7)']).to_string().replace('\n', '\n  '))
    
    print("\n" + "-" * 60)
    print("COMBINED: Premium (>=7) vs Non-Premium (<7)")
    print("-" * 60)
    print(f"  Accuracy:  {combined['accuracy']:.4f}")
    print(f"  F1-Score:  {combined['f1']:.4f}")
    print(f"  AUC-ROC:   {combined['auc_roc']:.4f}")
    print(f"  Precision: {combined['precision']:.4f}")
    print(f"  Recall:    {combined['recall']:.4f}")
    print("\n  Confusion Matrix:")
    print(pd.DataFrame(combined['confusion_matrix'],
        index=['Non-Premium', 'Premium'],
        columns=['Non-Premium', 'Premium']).to_string().replace('\n', '\n  '))
    
    return experiment_data


if __name__ == "__main__":
    main()
