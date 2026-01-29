"""
Hierarchical Binary Classification for Wine Quality.
Two-stage approach: Stage 1 classifies Better vs Regular, Stage 2 classifies Premium vs Good.

================================================================================
RESULTS SUMMARY
================================================================================

STAGE 1: Better (>=6) vs Regular (<6)
  - Accuracy:      [TO BE UPDATED]
  - F1-Score:      [TO BE UPDATED]
  - AUC-ROC:       [TO BE UPDATED]

STAGE 2: Premium (>=7) vs Good (=6)
  - Accuracy:      [TO BE UPDATED]
  - F1-Score:      [TO BE UPDATED]
  - AUC-ROC:       [TO BE UPDATED]

COMBINED: Premium (>=7) vs Non-Premium (<7)
  - Accuracy:      [TO BE UPDATED]
  - F1-Score:      [TO BE UPDATED]  ← Compare with previous best (0.6667)
  - AUC-ROC:       [TO BE UPDATED]

================================================================================

APPROACH:
Hierarchical/Cascaded Classification to address severe class imbalance (13.7% Premium).

Stage 1: Better vs Regular
  - Target: quality >= 6 vs quality < 6
  - Expected balance: ~55% Better, ~45% Regular (well balanced!)

Stage 2: Premium vs Good (only for "Better" wines)
  - Target: quality >= 7 vs quality = 6
  - Expected balance: ~25-30% Premium, ~70-75% Good (better than 13.7%!)

Combined Prediction Logic:
  - Premium: Stage1=Better AND Stage2=Premium
  - Non-Premium: Stage1=Regular OR (Stage1=Better AND Stage2=Good)

Preprocessing Pipeline (same for both stages):
1. Handle missing values and duplicates
2. Remove multicollinear features (correlation > 0.7)
3. Stratified train/test split (80/20) - based on original quality
4. Yeo-Johnson transformation (fit on train)
5. RobustScaler (fit on train)
6. class_weight='balanced' for both models

Model: Logistic Regression (both stages)
"""
import sys
import os
from typing import Tuple, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score
)

from preprocessing import (
    load_wine_data,
    handle_missing_values,
    remove_duplicates,
    remove_multicollinear_features,
    apply_yeo_johnson_transform,
    apply_robust_scaling
)
from experiments.experiment_tracker import save_experiment_results

# Experiment configuration
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORRELATION_THRESHOLD = 0.7
N_ITER_SEARCH = 30  # Number of parameter combinations to try
CV_FOLDS = 5


def prepare_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Prepare dataset: clean data, remove multicollinearity.
    Does NOT create binary target - that's done separately for each stage.
    """
    # Keep original quality column for target creation
    if verbose:
        print("\nOriginal quality distribution:")
        print(df['quality'].value_counts().sort_index())
    
    # Handle missing values and duplicates
    df = handle_missing_values(df, strategy='drop')
    df = remove_duplicates(df)
    
    if verbose:
        print(f"\nData shape after cleaning: {df.shape}")
    
    return df


def remove_multicollinearity_for_target(
    df: pd.DataFrame, 
    target_column: str, 
    verbose: bool = True
) -> Tuple[pd.DataFrame, list]:
    """Remove multicollinear features based on target correlation."""
    if verbose:
        print(f"\nRemoving multicollinear features (threshold > {CORRELATION_THRESHOLD}):")
    
    df, removed_features = remove_multicollinear_features(
        df,
        target_column=target_column,
        threshold=CORRELATION_THRESHOLD,
        verbose=verbose
    )
    
    return df, removed_features


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    verbose: bool = True
) -> float:
    """Find optimal decision threshold to maximize F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    
    if verbose:
        print(f"    Optimal threshold: {optimal_threshold:.4f}")
    
    return optimal_threshold


def train_stage_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    stage_name: str,
    class_names: list,
    verbose: bool = True
) -> Dict:
    """Train a single stage model with hyperparameter tuning."""
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"{stage_name}")
        print("=" * 60)
        
        # Show class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            print(f"  {class_names[cls]} ({cls}): {count} samples ({pct:.2f}%)")
    
    # Parameter distributions
    param_distributions = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'class_weight': ['balanced']
    }
    
    base_model = LogisticRegression(
        solver='saga',
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"\nHyperparameter tuning ({N_ITER_SEARCH} combinations, {CV_FOLDS}-fold CV)...")
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    if verbose:
        print(f"Best params: {random_search.best_params_}")
        print(f"Best CV F1: {random_search.best_score_:.4f}")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
    
    # Metrics with default threshold
    accuracy_default = accuracy_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Find optimal threshold
    if verbose:
        print(f"\nThreshold optimization:")
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, verbose=verbose)
    
    # Apply optimal threshold
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    
    if verbose:
        print(f"\n--- {stage_name} Results ---")
        print(f"Accuracy (default):   {accuracy_default:.4f}")
        print(f"Accuracy (optimized): {accuracy_optimized:.4f}")
        print(f"F1-Score (default):   {f1_default:.4f}")
        print(f"F1-Score (optimized): {f1_optimized:.4f}")
        print(f"AUC-ROC:              {auc_roc:.4f}")
        print(f"Precision:            {precision:.4f}")
        print(f"Recall:               {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': model,
        'accuracy': accuracy_optimized,
        'accuracy_default': accuracy_default,
        'f1': f1_optimized,
        'f1_default': f1_default,
        'auc_roc': auc_roc,
        'precision': precision,
        'recall': recall,
        'optimal_threshold': optimal_threshold,
        'y_test': y_test,
        'y_pred': y_pred_optimized,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def predict_hierarchical(
    X: np.ndarray,
    stage1_model,
    stage2_model,
    stage1_threshold: float,
    stage2_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make hierarchical predictions.
    
    Returns:
        final_pred: Binary predictions (1=Premium, 0=Non-Premium)
        final_proba: Combined probability of being Premium
    """
    # Stage 1: Better vs Regular
    stage1_proba = stage1_model.predict_proba(X)[:, 1]
    stage1_pred = (stage1_proba >= stage1_threshold).astype(int)
    
    # Initialize final predictions
    final_pred = np.zeros(len(X), dtype=int)
    final_proba = np.zeros(len(X))
    
    # Stage 2: Only for samples predicted as Better
    better_mask = stage1_pred == 1
    
    if better_mask.sum() > 0:
        stage2_proba = stage2_model.predict_proba(X[better_mask])[:, 1]
        stage2_pred = (stage2_proba >= stage2_threshold).astype(int)
        
        # Final predictions for "Better" samples
        final_pred[better_mask] = stage2_pred
        
        # Combined probability: P(Premium) = P(Better) × P(Premium|Better)
        final_proba[better_mask] = stage1_proba[better_mask] * stage2_proba
    
    # For "Regular" samples: final_pred stays 0, final_proba stays 0
    
    return final_pred, final_proba


def evaluate_combined(
    y_true_premium: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    verbose: bool = True
) -> Dict:
    """Evaluate combined hierarchical predictions against original Premium target."""
    
    accuracy = accuracy_score(y_true_premium, y_pred)
    f1 = f1_score(y_true_premium, y_pred)
    precision = precision_score(y_true_premium, y_pred)
    recall = recall_score(y_true_premium, y_pred)
    
    # AUC-ROC using combined probabilities
    auc_roc = roc_auc_score(y_true_premium, y_pred_proba)
    
    # Find optimal threshold on combined probabilities
    if verbose:
        print(f"\nCombined threshold optimization:")
    optimal_threshold = find_optimal_threshold(y_true_premium, y_pred_proba, verbose=verbose)
    
    # Apply optimal threshold
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    accuracy_optimized = accuracy_score(y_true_premium, y_pred_optimized)
    f1_optimized = f1_score(y_true_premium, y_pred_optimized)
    precision_optimized = precision_score(y_true_premium, y_pred_optimized)
    recall_optimized = recall_score(y_true_premium, y_pred_optimized)
    
    cm = confusion_matrix(y_true_premium, y_pred_optimized)
    
    return {
        'accuracy': accuracy_optimized,
        'accuracy_default': accuracy,
        'f1': f1_optimized,
        'f1_default': f1,
        'auc_roc': auc_roc,
        'precision': precision_optimized,
        'recall': recall_optimized,
        'optimal_threshold': optimal_threshold,
        'y_pred': y_pred_optimized,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'class_names': ['Non-Premium', 'Premium']
    }


def plot_all_confusion_matrices(stage1_results, stage2_results, combined_results, 
                                 output_dir, experiment_name):
    """Plot all three confusion matrices in a single figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Confusion Matrices - {experiment_name}', fontsize=14, fontweight='bold')
    
    # Stage 1
    sns.heatmap(stage1_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=stage1_results['class_names'], 
                yticklabels=stage1_results['class_names'], ax=axes[0])
    axes[0].set_title(f"Stage 1: Better vs Regular\nF1={stage1_results['f1']:.4f}, Acc={stage1_results['accuracy']:.4f}")
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Stage 2
    sns.heatmap(stage2_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=stage2_results['class_names'],
                yticklabels=stage2_results['class_names'], ax=axes[1])
    axes[1].set_title(f"Stage 2: Premium vs Good\nF1={stage2_results['f1']:.4f}, Acc={stage2_results['accuracy']:.4f}")
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # Combined
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
    """Run the full hierarchical classification experiment."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    if verbose:
        print("=" * 60)
        print(f"EXPERIMENT: {EXPERIMENT_NAME}")
        print("=" * 60)
        print("\nHierarchical Classification Approach:")
        print("  Stage 1: Better (>=6) vs Regular (<6)")
        print("  Stage 2: Premium (>=7) vs Good (=6)")
        print("\n[Step 1] Loading red wine data...")
    
    df = load_wine_data(data_path, wine_type='red')
    
    if verbose:
        print(f"Original data shape: {df.shape}")
        print("\n[Step 2] Cleaning data...")
    
    df = prepare_data(df.copy(), verbose=verbose)
    
    # Create target variables for multicollinearity removal
    df['is_premium'] = (df['quality'] >= 7).astype(int)  # Original target (for combined eval)
    
    if verbose:
        print("\n[Step 3] Target distributions:")
        better_count = (df['quality'] >= 6).sum()
        print(f"  Better (>=6):  {better_count} / {len(df)} ({better_count/len(df)*100:.1f}%)")
        print(f"  Premium (>=7): {df['is_premium'].sum()} / {len(df)} ({df['is_premium'].mean()*100:.1f}%)")
    
    # Remove multicollinearity based on is_premium (original target)
    df_clean, removed_features = remove_multicollinearity_for_target(
        df, target_column='is_premium', verbose=verbose
    )
    
    # Now create all target variables on the cleaned dataframe
    df_clean['is_better'] = (df_clean['quality'] >= 6).astype(int)  # Stage 1 target
    df_clean['is_premium'] = (df_clean['quality'] >= 7).astype(int)  # Original target
    
    # Feature columns (excluding all targets)
    target_cols = ['quality', 'is_better', 'is_premium']
    feature_cols = [col for col in df_clean.columns if col not in target_cols]
    
    if verbose:
        print(f"\nFeatures: {feature_cols}")
    
    # Prepare data
    X = df_clean[feature_cols].values
    y_better = df_clean['is_better'].values  # Stage 1 target
    y_premium = df_clean['is_premium'].values  # Original target
    quality = df_clean['quality'].values  # Keep for Stage 2 filtering
    
    # =========================================================================
    # STRATIFIED SPLIT - Stratify by ORIGINAL quality to maintain distribution
    # =========================================================================
    if verbose:
        print(f"\n[Step 4] Stratified train/test split (80/20)...")
    
    # Split indices to keep all targets aligned
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=quality,  # Stratify by quality to maintain class proportions
        random_state=RANDOM_STATE
    )
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_better_train, y_better_test = y_better[train_idx], y_better[test_idx]
    y_premium_train, y_premium_test = y_premium[train_idx], y_premium[test_idx]
    quality_train, quality_test = quality[train_idx], quality[test_idx]
    
    if verbose:
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        print("\n[Step 5] Applying Yeo-Johnson transformation...")
    
    X_train, X_test, yj_transformer = apply_yeo_johnson_transform(
        X_train, X_test, feature_cols, verbose=verbose
    )
    
    if verbose:
        print("\n[Step 6] Applying RobustScaler...")
    
    X_train, X_test, scaler = apply_robust_scaling(
        X_train, X_test, verbose=verbose
    )
    
    # =========================================================================
    # STAGE 1: Better (>=6) vs Regular (<6)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("[Step 7] TRAINING STAGE 1: Better (>=6) vs Regular (<6)")
        print("=" * 60)
    
    stage1_results = train_stage_model(
        X_train, X_test,
        y_better_train, y_better_test,
        stage_name="STAGE 1: Better vs Regular",
        class_names=['Regular (<6)', 'Better (>=6)'],
        verbose=verbose
    )
    
    # =========================================================================
    # STAGE 2: Premium (>=7) vs Good (=6) - ONLY for "Better" wines
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("[Step 8] TRAINING STAGE 2: Premium (>=7) vs Good (=6)")
        print("=" * 60)
        print("  (Training only on wines with quality >= 6)")
    
    # Filter training data to only "Better" wines (quality >= 6)
    better_train_mask = quality_train >= 6
    X_train_better = X_train[better_train_mask]
    quality_train_better = quality_train[better_train_mask]
    
    # Stage 2 target: Premium (>=7) vs Good (=6)
    y_premium_among_better_train = (quality_train_better >= 7).astype(int)
    
    # For test set, also filter to "Better" wines for Stage 2 evaluation
    better_test_mask = quality_test >= 6
    X_test_better = X_test[better_test_mask]
    quality_test_better = quality_test[better_test_mask]
    y_premium_among_better_test = (quality_test_better >= 7).astype(int)
    
    if verbose:
        print(f"\n  Stage 2 training samples: {len(X_train_better)}")
        print(f"  Stage 2 test samples: {len(X_test_better)}")
    
    stage2_results = train_stage_model(
        X_train_better, X_test_better,
        y_premium_among_better_train, y_premium_among_better_test,
        stage_name="STAGE 2: Premium vs Good",
        class_names=['Good (=6)', 'Premium (>=7)'],
        verbose=verbose
    )
    
    # =========================================================================
    # COMBINED EVALUATION: Map back to original Premium vs Non-Premium
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("[Step 9] COMBINED EVALUATION: Premium (>=7) vs Non-Premium (<7)")
        print("=" * 60)
        print("  Evaluating hierarchical predictions on FULL test set")
    
    # Make hierarchical predictions on full test set
    combined_pred, combined_proba = predict_hierarchical(
        X_test,
        stage1_results['model'],
        stage2_results['model'],
        stage1_results['optimal_threshold'],
        stage2_results['optimal_threshold']
    )
    
    # Evaluate against original premium target
    combined_results = evaluate_combined(
        y_premium_test,
        combined_pred,
        combined_proba,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n{'=' * 60}")
        print("COMBINED RESULTS (Premium vs Non-Premium)")
        print("=" * 60)
        print(f"Accuracy (default):   {combined_results['accuracy_default']:.4f}")
        print(f"Accuracy (optimized): {combined_results['accuracy']:.4f}")
        print(f"F1-Score (default):   {combined_results['f1_default']:.4f}")
        print(f"F1-Score (optimized): {combined_results['f1']:.4f}")
        print(f"AUC-ROC:              {combined_results['auc_roc']:.4f}")
        print(f"Precision:            {combined_results['precision']:.4f}")
        print(f"Recall:               {combined_results['recall']:.4f}")
        
        print("\nClassification Report (Combined):")
        print(classification_report(
            y_premium_test, 
            combined_results['y_pred'],
            target_names=['Non-Premium', 'Premium']
        ))
        
        print("Confusion Matrix (Combined):")
        print(pd.DataFrame(
            combined_results['confusion_matrix'],
            index=['Non-Premium', 'Premium'],
            columns=['Non-Premium', 'Premium']
        ))
    
    # Store all results
    return {
        'experiment_name': EXPERIMENT_NAME,
        'feature_names': feature_cols,
        'is_binary': True,
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
        'combined_results': combined_results,
        'y_premium_test': y_premium_test,
        'X_test': X_test,
        'yj_transformer': yj_transformer,
        'scaler': scaler
    }


def main():
    """Main entry point."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')
    
    # Run experiment
    experiment_data = run_experiment(verbose=True)
    
    stage1 = experiment_data['stage1_results']
    stage2 = experiment_data['stage2_results']
    combined = experiment_data['combined_results']
    
    # Save confusion matrix
    print("\n[Step 10] Saving confusion matrix...")
    plot_all_confusion_matrices(stage1, stage2, combined, output_dir, EXPERIMENT_NAME)
    
    # Save to CSV tracker
    print("\n[Step 11] Updating experiments tracker...")
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={
            'accuracy': combined['accuracy'],
            'f1': combined['f1'],
            'auc_roc': combined['auc_roc']
        },
        output_dir=output_dir
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLASSIFICATION COMPLETE")
    print("=" * 60)
    
    print("\n" + "-" * 60)
    print("STAGE 1: Better (>=6) vs Regular (<6)")
    print("-" * 60)
    print(f"  Accuracy:  {stage1['accuracy']:.4f}")
    print(f"  F1-Score:  {stage1['f1']:.4f}")
    print(f"  AUC-ROC:   {stage1['auc_roc']:.4f}")
    print(f"  Precision: {stage1['precision']:.4f}")
    print(f"  Recall:    {stage1['recall']:.4f}")
    print(f"  Threshold: {stage1['optimal_threshold']:.4f}")
    print("\n  Confusion Matrix (Stage 1):")
    print(pd.DataFrame(
        stage1['confusion_matrix'],
        index=['Regular (<6)', 'Better (>=6)'],
        columns=['Regular (<6)', 'Better (>=6)']
    ).to_string().replace('\n', '\n  '))
    
    print("\n" + "-" * 60)
    print("STAGE 2: Premium (>=7) vs Good (=6)")
    print("-" * 60)
    print(f"  Accuracy:  {stage2['accuracy']:.4f}")
    print(f"  F1-Score:  {stage2['f1']:.4f}")
    print(f"  AUC-ROC:   {stage2['auc_roc']:.4f}")
    print(f"  Precision: {stage2['precision']:.4f}")
    print(f"  Recall:    {stage2['recall']:.4f}")
    print(f"  Threshold: {stage2['optimal_threshold']:.4f}")
    print("\n  Confusion Matrix (Stage 2):")
    print(pd.DataFrame(
        stage2['confusion_matrix'],
        index=['Good (=6)', 'Premium (>=7)'],
        columns=['Good (=6)', 'Premium (>=7)']
    ).to_string().replace('\n', '\n  '))
    
    print("\n" + "-" * 60)
    print("COMBINED: Premium (>=7) vs Non-Premium (<7)")
    print("-" * 60)
    print(f"  Accuracy:  {combined['accuracy']:.4f}")
    print(f"  F1-Score:  {combined['f1']:.4f}  ← Previous best: 0.6667")
    print(f"  AUC-ROC:   {combined['auc_roc']:.4f}")
    print(f"  Precision: {combined['precision']:.4f}")
    print(f"  Recall:    {combined['recall']:.4f}")
    print(f"  Threshold: {combined['optimal_threshold']:.4f}")
    
    improvement = combined['f1'] - 0.6667
    if improvement > 0:
        print(f"\n  IMPROVEMENT over baseline: +{improvement:.4f}")
    else:
        print(f"\n  Change from baseline: {improvement:.4f}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_data = main()
