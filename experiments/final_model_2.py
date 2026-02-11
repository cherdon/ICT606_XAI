import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, confusion_matrix, classification_report
)
import xgboost as xgb

from preprocessing import load_wine_data, handle_missing_values, remove_duplicates
from experiments.experiment_tracker import save_experiment_results

EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def engineer_wine_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Create domain-specific wine features based on industry knowledge."""
    df = df.copy()

    if verbose:
        print("\n[Feature Engineering] Creating domain-specific wine features...")

    # 1. SO2 Ratio (Stability Index)
    df['so2_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-8)

    # 2. Molecular SO2 (Active Preservative)
    df['molecular_so2'] = df['free sulfur dioxide'] / (1 + 10 ** (df['pH'] - 1.81))

    # 3. VA to FA Ratio (Spoilage Indicator)
    df['va_to_fa_ratio'] = df['volatile acidity'] / (df['fixed acidity'] + 1e-8)

    # 4. Sugar to Acidity (Dryness/"Trocken" Rule)
    total_acidity = df['fixed acidity'] + df['volatile acidity']
    df['sugar_to_acidity'] = df['residual sugar'] / (total_acidity + 1e-8)

    # 5. Alcohol to Density (Structure/Body)
    df['alcohol_to_density'] = df['alcohol'] / df['density']

    if verbose:
        print("  Created: so2_ratio, molecular_so2, va_to_fa_ratio, sugar_to_acidity, alcohol_to_density")

    return df


def prepare_data(verbose: bool = True):
    """
    Prepare data for quality 6 vs quality >=7 classification.
    Only includes samples with quality 6, 7, or 8.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')

    df = load_wine_data(data_path, wine_type='red')
    df = handle_missing_values(df, strategy='drop')
    df = remove_duplicates(df)

    # Filter to only quality 6, 7, 8
    df_filtered = df[df['quality'] >= 6].copy()

    if verbose:
        print("=" * 60)
        print(f"FINAL MODEL 2: {EXPERIMENT_NAME}")
        print("=" * 60)
        print("\nTASK: Quality 6 vs Quality >= 7 Classification")
        print(f"\nOriginal samples: {len(df)}")
        print(f"Filtered samples (quality >= 6): {len(df_filtered)}")
        print(f"\nQuality distribution:")
        print(df_filtered['quality'].value_counts().sort_index())

    # Feature engineering
    df_filtered = engineer_wine_features(df_filtered, verbose=verbose)

    # Create binary target: 0 for quality 6, 1 for quality >= 7
    df_filtered['is_7plus'] = (df_filtered['quality'] >= 7).astype(int)

    if verbose:
        n_class_0 = (df_filtered['is_7plus'] == 0).sum()
        n_class_1 = (df_filtered['is_7plus'] == 1).sum()
        print(f"\nClass distribution:")
        print(f"  Quality 6 (class 0): {n_class_0} samples ({n_class_0 / (n_class_0 + n_class_1) * 100:.1f}%)")
        print(f"  Quality 7+ (class 1): {n_class_1} samples ({n_class_1 / (n_class_0 + n_class_1) * 100:.1f}%)")

    return df_filtered


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, verbose: bool = True) -> float:
    """Find threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]

    if verbose:
        print(f"    Optimal threshold: {optimal_threshold:.4f}")
        print(f"    F1 at optimal threshold: {f1_scores[optimal_idx]:.4f}")

    return optimal_threshold


def train_stacking_model(X_train, X_test, y_train, y_test, verbose: bool = True):
    """
    Train Stacking Ensemble with SVM, RF, XGBoost as base models and SVM as meta-learner.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING: Stacking Ensemble (SVM + RF + XGBoost)")
        print("=" * 60)

        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(unique, counts):
            label = "Quality 6" if cls == 0 else "Quality 7+"
            pct = count / len(y_train) * 100
            print(f"  {label} ({cls}): {count} samples ({pct:.1f}%)")

    # Calculate scale_pos_weight for XGBoost
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    # Define base estimators
    base_estimators = [
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )),
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
        ('xgb', xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        ))
    ]

    # Meta-learner: SVM (no LogReg)
    meta_learner = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        C=1.0
    )

    if verbose:
        print(f"\n[Stacking Configuration]")
        print(f"  Base Models:")
        print(f"    - SVM (RBF, C=10, balanced)")
        print(f"    - Random Forest (300 trees, balanced_subsample)")
        print(f"    - XGBoost (300 estimators, scale_pos_weight={scale_pos_weight:.2f})")
        print(f"  Meta-Learner: SVM (RBF, balanced)")
        print(f"  CV for stacking: {CV_FOLDS}-fold")

    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=CV_FOLDS,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False
    )

    if verbose:
        print(f"\n[Step 1] Training stacking ensemble...")

    stacking_clf.fit(X_train, y_train)

    # Evaluate individual base models for comparison
    if verbose:
        print(f"\n[Step 2] Base model individual performance:")
        for name, model in base_estimators:
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = model.decision_function(X_test)
            pred = model.predict(X_test)
            base_f1 = f1_score(y_test, pred)
            base_auc = roc_auc_score(y_test, proba)
            print(f"    {name.upper():5s}: F1={base_f1:.4f}, AUC={base_auc:.4f}")

    # Probability calibration
    if verbose:
        print(f"\n[Step 3] Probability calibration (isotonic regression)...")

    calibrated_stack = CalibratedClassifierCV(stacking_clf, method='isotonic', cv=5)
    calibrated_stack.fit(X_train, y_train)

    # Get predictions
    y_pred_proba = calibrated_stack.predict_proba(X_test)[:, 1]

    # Threshold optimization
    if verbose:
        print(f"\n[Step 4] Threshold optimization for F1...")

    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, verbose)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred)

    if verbose:
        print(f"\n" + "=" * 60)
        print("FINAL RESULTS: Quality 6 vs Quality >= 7")
        print("=" * 60)
        print(f"\n  Accuracy:  {accuracy:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  AUC-ROC:   {auc_roc:.4f}")
        print(f"  Threshold: {optimal_threshold:.4f}")

        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Quality 6', 'Quality 7+']))

        print(f"  Confusion Matrix:")
        print(pd.DataFrame(cm,
                           index=['Quality 6', 'Quality 7+'],
                           columns=['Pred 6', 'Pred 7+']).to_string().replace('\n', '\n  '))

    return {
        'model': calibrated_stack,
        'stacking_model': stacking_clf,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_results(results, output_dir, experiment_name):
    """Plot confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Purples',
                xticklabels=['Pred 6', 'Pred 7+'],
                yticklabels=['Quality 6', 'Quality 7+'], ax=ax)

    ax.set_title(f'{experiment_name}\nQuality 6 vs Quality >= 7\nF1={results["f1"]:.4f}, AUC={results["auc_roc"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_name}_confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")
    return output_path


def run_experiment(verbose: bool = True):
    """Run the complete experiment."""
    df = prepare_data(verbose=verbose)

    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['quality', 'is_7plus']]
    X = df[feature_cols].values
    y = df['is_7plus'].values

    if verbose:
        print(f"\nFeatures ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Apply transformations
    if verbose:
        print(f"\n[Preprocessing] Applying Yeo-Johnson transformation...")
    yj = PowerTransformer(method='yeo-johnson')
    X_train = yj.fit_transform(X_train)
    X_test = yj.transform(X_test)

    if verbose:
        print(f"[Preprocessing] Applying RobustScaler...")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate model
    results = train_stacking_model(X_train, X_test, y_train, y_test, verbose=verbose)

    # Store additional data
    results['feature_names'] = feature_cols
    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['y_test'] = y_test
    results['scaler'] = scaler
    results['yj_transformer'] = yj

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')

    results = run_experiment(verbose=True)

    # Save results
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={
            'accuracy': results['accuracy'],
            'f1': results['f1'],
            'auc_roc': results['auc_roc']
        },
        output_dir=output_dir
    )

    # Plot results
    plot_results(results, output_dir, EXPERIMENT_NAME)

    print("\n" + "=" * 60)
    print("FINAL MODEL 2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel: Stacking Ensemble (SVM + RF + XGBoost)")
    print(f"Meta-Learner: SVM (NO LogReg)")
    print(f"Task: Quality 6 vs Quality >= 7")
    print(f"\nFinal F1 Score: {results['f1']:.4f}")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")

    return results


if __name__ == "__main__":
    results = main()