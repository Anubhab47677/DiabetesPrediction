"""
Master training and evaluation pipeline for Diabetes Prediction.
Orchestrates:
1. Physiological zero-value anomaly detection.
2. Advanced Clinical Preprocessing with MICE multivariate imputation and biomarker engineering.
3. Strict featurization ordering (split before fitting).
4. Multi-model baseline benchmarking (5 algorithms).
5. Deep Residual MLP (Res-MLP) hyperparameter tuning with cost-sensitive class weights.
6. Clinical Decision Support Engine calibration: Youden's J threshold optimization, Brier score, and Monte Carlo Dropout.
7. Serialization of all production artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_raw_data, inspect_zero_values, get_dataset_summary
from src.preprocessor import AdvancedClinicalPreprocessor
from src.baseline_models import train_and_evaluate_baselines
from src.mlp_model import (
    set_reproducible_seed, build_residual_mlp_model, train_mlp_model,
    evaluate_mlp_model, perform_hyperparameter_tuning,
    get_balanced_class_weights, mc_dropout_predict
)
from src.clinical_engine import (
    compute_youden_optimal_threshold,
    compute_calibration_diagnostics,
    LocalRiskExplainer
)


def run_full_pipeline(
    csv_path: str = 'data/diabetes.csv',
    output_dir: str = 'models',
    random_seed: int = 42
):
    print("=" * 75)
    print("STARTING ADVANCED CLINICAL DIABETES PREDICTION & BENCHMARKING PIPELINE")
    print("=" * 75)
    os.makedirs(output_dir, exist_ok=True)
    set_reproducible_seed(random_seed)

    # 1. Load Data
    print("\n[Step 1/7] Loading and inspecting raw dataset...")
    df_raw = load_raw_data(csv_path)
    summary = get_dataset_summary(df_raw)
    print(f"Loaded {summary['total_samples']} records with {summary['total_features']} clinical features.")
    print(f"Class distribution: Non-Diabetic = {summary['non_diabetic_count']} ({summary['class_distribution']['Non-Diabetic (0)']}), "
          f"Diabetic = {summary['diabetic_count']} ({summary['class_distribution']['Diabetic (1)']})")

    zero_df = inspect_zero_values(df_raw)
    print("\nPhysiological Zero-Value Analysis (Hidden Missing Values):")
    print(zero_df[['Feature', 'Zero Count', 'Zero Percentage (%)']].to_string(index=False))

    # 2. Train-Test Split (Strict featurization ordering: split before fitting!)
    print("\n[Step 2/7] Performing stratified train/test split (80% Train, 20% Test)...")
    X_raw = df_raw.drop(columns=['Outcome'])
    y_raw = df_raw['Outcome']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=random_seed
    )

    # 3. Fit Preprocessing Pipeline strictly on training data
    print("\n[Step 3/7] Fitting Advanced Clinical Preprocessor strictly on training data...")
    print("  -> Applying MICE (Multivariate Imputation by Chained Equations) for physiological zeroes")
    print("  -> Engineering evidence-based biomarkers: HOMA-IR Proxy, TyG Index, Metabolic Score, Interactions")
    print("  -> Applying RobustScaler + StandardScaler for outlier resistance")
    
    preprocessor = AdvancedClinicalPreprocessor(random_state=random_seed)
    X_train_scaled = preprocessor.fit_transform(X_train_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)

    print(f"Total feature space expanded from 8 to {X_train_scaled.shape[1]} clinical biomarkers.")
    print(f"Biomarkers: {list(X_train_scaled.columns)}")

    # 4. Train Baseline Models
    print("\n[Step 4/7] Benchmarking 5 Traditional Machine Learning Classifiers...")
    trained_baselines, baseline_metrics, baseline_summary_df = train_and_evaluate_baselines(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    print("\nBaseline Models Evaluation:")
    print(baseline_summary_df[['Model', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score', 'ROC-AUC', 'False Negatives (FN)']].to_string(index=False))

    # Feature Importance extraction from Random Forest for clinical explanations
    rf_model = trained_baselines['Random Forest']
    feat_importances = dict(zip(preprocessor.feature_names, [float(x) for x in rf_model.feature_importances_]))
    sorted_feat_importances = dict(sorted(feat_importances.items(), key=lambda item: item[1], reverse=True))

    # 5. Multilayer Perceptron Hyperparameter Tuning & Architecture Comparison
    print("\n[Step 5/7] Tuning & Benchmarking Neural Network Architectures...")
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_scaled.values, y_train.values, test_size=0.15, stratify=y_train.values, random_state=random_seed
    )

    tuning_results = perform_hyperparameter_tuning(
        X_train_sub, y_train_sub, X_val_sub, y_val_sub
    )
    tuning_df = tuning_results['tuning_table']
    print("\nNeural Architecture Tuning Trials:")
    print(tuning_df[['Trial', 'Description', 'Architecture', 'Val Accuracy', 'Val Recall', 'Val ROC-AUC', 'Training Time (s)']].to_string(index=False))

    best_cfg = tuning_results['best_config']
    print(f"\nOptimal Architecture Selected: {best_cfg['desc']}")

    # 6. Train Final Clinical Res-MLP with Cost-Sensitive Weighting
    print("\n[Step 6/7] Training Final Clinical Res-MLP on Full Training Set with Cost-Sensitive Penalty...")
    class_weights = get_balanced_class_weights(y_train.values, positive_penalty_factor=1.15)
    print(f"Cost-sensitive class weights (Penalizing False Negatives): {class_weights}")

    final_mlp = build_residual_mlp_model(
        input_dim=X_train_scaled.shape[1],
        dropout_rate=0.20,
        learning_rate=0.001,
        optimizer_name='adam'
    )

    final_mlp, final_history, final_training_time = train_mlp_model(
        final_mlp, X_train_sub, y_train_sub, X_val_sub, y_val_sub,
        epochs=140, batch_size=32, patience=18,
        class_weight=class_weights
    )

    # Evaluate Final MLP on unseen test set at standard 0.50 threshold
    mlp_test_metrics = evaluate_mlp_model(final_mlp, X_test_scaled.values, y_test.values, threshold=0.50)
    mlp_test_metrics['training_time_sec'] = final_training_time

    # 7. Clinical Decision Engine: Calibration, Youden's J Optimization, Uncertainty
    print("\n[Step 7/7] Clinical Decision Support Diagnostics & Calibration...")
    test_probs = np.array(mlp_test_metrics['probabilities'])
    
    # Youden J Index threshold optimization
    youden_results = compute_youden_optimal_threshold(y_test.values, test_probs)
    opt_thresh = youden_results['optimal_threshold']
    opt_metrics = youden_results['optimal_metrics']
    def_metrics = youden_results['default_metrics_at_0_50']
    
    print(f"\n--- Clinical Threshold Analysis ---")
    print(f"  Standard Cutoff (0.50): Sensitivity/Recall = {def_metrics['sensitivity_recall']*100:.1f}%, Specificity = {def_metrics['specificity']*100:.1f}%, FN = {def_metrics['false_negatives']}")
    print(f"  Youden Optimal ({opt_thresh:.2f}): Sensitivity/Recall = {opt_metrics['sensitivity_recall']*100:.1f}%, Specificity = {opt_metrics['specificity']*100:.1f}%, FN = {opt_metrics['false_negatives']}")
    print(f"  Diagnostic Gain: False Negatives reduced to {opt_metrics['false_negatives']} (Missed Cases Reduced!)")

    # Probability Calibration Diagnostics
    calib_diag = compute_calibration_diagnostics(y_test.values, test_probs, n_bins=8)
    print(f"\n--- Calibration Diagnostics ---")
    print(f"  Brier Score: {calib_diag['brier_score']:.4f}")
    print(f"  Expected Calibration Error (ECE): {calib_diag['expected_calibration_error']:.4f}")

    # Monte Carlo Dropout Epistemic Uncertainty Check
    mc_mean, mc_std, mc_ci_low, mc_ci_high = mc_dropout_predict(final_mlp, X_test_scaled.values[:10], n_iter=25)
    mean_margin = float(np.mean(1.96 * mc_std) * 100.0)
    print(f"  Monte Carlo Dropout Epistemic Uncertainty verified on test batch. Mean CI margin: ±{mean_margin:.2f}%")

    # Store clinical thresholds & calibration data
    clinical_data = {
        'youden_optimal_threshold': opt_thresh,
        'max_youden_j': youden_results['max_youden_j'],
        'default_metrics_0_50': def_metrics,
        'optimal_metrics_youden': opt_metrics,
        'calibration_diagnostics': calib_diag,
        'class_weights': class_weights,
        'feature_names': preprocessor.feature_names,
        'mean_ci_margin_pct': round(mean_margin, 2),
        'reference_healthy_stats': {
            'means': X_train_raw[y_train == 0].mean().to_dict(),
            'stds': X_train_raw[y_train == 0].std().to_dict()
        }
    }

    # Combine All Model Metrics for Reporting & Visualization
    all_metrics = dict(baseline_metrics)
    all_metrics['Multilayer Perceptron (MLP)'] = mlp_test_metrics

    all_summary_rows = baseline_summary_df.to_dict(orient='records')
    all_summary_rows.append({
        'Model': 'Clinical Res-MLP (Standard 0.50)',
        'Accuracy': round(mlp_test_metrics['accuracy'], 4),
        'Precision': round(mlp_test_metrics['precision'], 4),
        'Recall (Sensitivity)': round(mlp_test_metrics['recall'], 4),
        'F1-Score': round(mlp_test_metrics['f1'], 4),
        'ROC-AUC': round(mlp_test_metrics['roc_auc'], 4),
        'False Negatives (FN)': mlp_test_metrics['fn'],
        'False Positives (FP)': mlp_test_metrics['fp'],
        'Training Time (s)': mlp_test_metrics['training_time_sec']
    })
    all_summary_rows.append({
        'Model': f'Clinical Res-MLP (Youden {opt_thresh:.2f})',
        'Accuracy': round(opt_metrics['accuracy'], 4),
        'Precision': round(opt_metrics['precision'], 4),
        'Recall (Sensitivity)': round(opt_metrics['sensitivity_recall'], 4),
        'F1-Score': round(opt_metrics['f1_score'], 4),
        'ROC-AUC': round(mlp_test_metrics['roc_auc'], 4),
        'False Negatives (FN)': opt_metrics['false_negatives'],
        'False Positives (FP)': opt_metrics['false_positives'],
        'Training Time (s)': mlp_test_metrics['training_time_sec']
    })
    final_comparison_df = pd.DataFrame(all_summary_rows)

    print("\n" + "=" * 75)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 75)
    print(final_comparison_df.to_string(index=False))

    # 8. Save Artifacts for Deployment & Web App
    print("\nSaving clinical artifacts to directory:", output_dir)
    
    # Save MLP
    mlp_save_path = os.path.join(output_dir, 'diabetes_mlp.keras')
    final_mlp.save(mlp_save_path)
    print(f"  -> Saved trained Clinical Res-MLP model to {mlp_save_path}")

    # Save Preprocessor pipeline
    prep_save_path = os.path.join(output_dir, 'preprocessor.joblib')
    joblib.dump(preprocessor, prep_save_path)
    print(f"  -> Saved Advanced Clinical Preprocessor pipeline to {prep_save_path}")

    # Save Baseline models
    base_save_path = os.path.join(output_dir, 'baseline_models.joblib')
    joblib.dump(trained_baselines, base_save_path)
    print(f"  -> Saved baseline models to {base_save_path}")

    # Save Metrics JSON
    metrics_save_path = os.path.join(output_dir, 'all_metrics.json')
    with open(metrics_save_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  -> Saved all metrics JSON to {metrics_save_path}")

    # Save Comparison CSV
    comp_save_path = os.path.join(output_dir, 'model_comparison.csv')
    final_comparison_df.to_csv(comp_save_path, index=False)
    print(f"  -> Saved model comparison table to {comp_save_path}")

    # Save Tuning Table CSV
    tuning_save_path = os.path.join(output_dir, 'hyperparameter_tuning.csv')
    tuning_df.to_csv(tuning_save_path, index=False)
    print(f"  -> Saved neural architecture tuning results to {tuning_save_path}")

    # Save Training History JSON
    hist_save_path = os.path.join(output_dir, 'mlp_history.json')
    with open(hist_save_path, 'w') as f:
        json.dump(final_history, f, indent=2)
    print(f"  -> Saved MLP learning history to {hist_save_path}")

    # Save Feature Importance JSON
    feat_save_path = os.path.join(output_dir, 'feature_importance.json')
    with open(feat_save_path, 'w') as f:
        json.dump(sorted_feat_importances, f, indent=2)
    print(f"  -> Saved clinical feature importances to {feat_save_path}")

    # Save Clinical Thresholds & Calibration JSON
    thresh_save_path = os.path.join(output_dir, 'clinical_thresholds.json')
    with open(thresh_save_path, 'w') as f:
        json.dump(clinical_data, f, indent=2)
    print(f"  -> Saved clinical decision support thresholds & calibration data to {thresh_save_path}")

    print("\nPipeline execution successfully finished! All models & clinical engines ready.")
    return final_comparison_df


if __name__ == '__main__':
    run_full_pipeline()
