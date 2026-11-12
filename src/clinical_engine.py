"""
Clinical Decision Support Engine for Diabetes Prediction.
Provides:
1. Youden's J Index threshold optimization for clinical screening.
2. Probability Calibration assessment & Reliability Curves (Brier Score, ECE).
3. Explainable Local Biomarker Risk Attribution (patient-level waterfall drivers).
4. Automated clinical screening interpretation and evidence-based recommendation logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    brier_score_loss, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve


def compute_youden_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Computes the Youden's J Statistic across all ROC operating thresholds:
      J = Sensitivity + Specificity - 1 = TPR - FPR
    Identifies the threshold that mathematically balances diagnostic sensitivity and specificity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])

    # Clamp threshold within reasonable clinical bounds [0.10, 0.90]
    optimal_threshold = float(np.clip(optimal_threshold, 0.15, 0.85))

    # Evaluate performance at default 0.50 vs optimal Youden threshold
    def get_metrics_at_thresh(thresh: float) -> Dict[str, Any]:
        preds = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        return {
            'threshold': round(thresh, 4),
            'sensitivity_recall': round(sens, 4),
            'specificity': round(spec),
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'f1_score': round(f1, 4),
            'false_negatives': int(fn),
            'false_positives': int(fp),
            'true_positives': int(tp),
            'true_negatives': int(tn)
        }

    default_metrics = get_metrics_at_thresh(0.50)
    optimal_metrics = get_metrics_at_thresh(optimal_threshold)

    return {
        'optimal_threshold': round(optimal_threshold, 4),
        'max_youden_j': round(float(j_scores[best_idx]), 4),
        'default_metrics_at_0_50': default_metrics,
        'optimal_metrics': optimal_metrics,
        'roc_curve_data': {
            'fpr': [float(x) for x in fpr],
            'tpr': [float(x) for x in tpr],
            'thresholds': [float(x) for x in thresholds],
            'j_scores': [float(x) for x in j_scores]
        }
    }


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 8
) -> Dict[str, Any]:
    """
    Computes reliability diagram data and Expected Calibration Error (ECE)
    to assess whether predicted probabilities correspond to empirical event frequencies.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    brier = brier_score_loss(y_true, y_prob)

    # Calculate Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    ece = 0.0
    total_samples = len(y_prob)

    for b in range(n_bins):
        in_bin = bin_assignments == b
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return {
        'brier_score': round(float(brier), 4),
        'expected_calibration_error': round(float(ece), 4),
        'calibration_curve': {
            'fraction_of_positives': [round(float(x), 4) for x in prob_true],
            'mean_predicted_value': [round(float(x), 4) for x in prob_pred]
        }
    }


class LocalRiskExplainer:
    """
    Computes local feature risk attribution for individual patient predictions
    by comparing patient biomarker levels against the healthy non-diabetic cohort baseline.
    """

    def __init__(
        self,
        reference_cohort: pd.DataFrame,
        feature_importances: Dict[str, float],
        feature_names: List[str]
    ):
        self.feature_names = feature_names
        self.feature_importances = feature_importances

        # Calculate healthy cohort reference baselines (mean & std)
        self.baseline_means = reference_cohort[feature_names].mean().to_dict()
        self.baseline_stds = reference_cohort[feature_names].std().replace(0, 1.0).to_dict()

    def explain_patient(
        self,
        patient_features: Dict[str, float],
        predicted_risk: float,
        top_n: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Computes the signed risk impact of each biomarker for the specific patient.
        Positive impact indicates the biomarker elevates diabetes risk above the healthy baseline.
        Negative impact indicates a protective factor.
        """
        attributions = []

        for feat in self.feature_names:
            if feat not in patient_features:
                continue

            val = float(patient_features[feat])
            mu = float(self.baseline_means.get(feat, 0.0))
            sigma = float(self.baseline_stds.get(feat, 1.0))
            importance = float(self.feature_importances.get(feat, 0.05))

            # Standardized z-score deviation from healthy baseline
            z_score = (val - mu) / sigma

            # Directional attribution weighted by feature importance
            risk_contribution = z_score * importance * 100.0

            # Friendly clinical names
            friendly_names = {
                'Glucose': 'Plasma Fasting Glucose',
                'BMI': 'Body Mass Index (BMI)',
                'HOMA_IR_Proxy': 'HOMA-IR Insulin Resistance Proxy',
                'TyG_Index': 'TyG Adiposity Surrogate Index',
                'Age': 'Patient Age',
                'DiabetesPedigreeFunction': 'Genetic Diabetes Pedigree',
                'Metabolic_Syndrome_Score': 'Metabolic Syndrome Score',
                'Insulin': 'Serum Fasting Insulin',
                'BloodPressure': 'Diastolic Blood Pressure',
                'Age_BMI_Risk_Interaction': 'Age-BMI Risk Synergism',
                'Glucose_BMI_Interaction': 'Glucose-BMI Co-Burden',
                'Pregnancies': 'Pregnancy History',
                'SkinThickness': 'Triceps Skinfold Thickness',
                'Log_Insulin': 'Log Serum Insulin',
                'BMI_Category': 'WHO Obesity Class',
                'Glucose_Category': 'ADA Glycemic Category',
                'Insulin_Glucose_Ratio': 'Insulin/Glucose Ratio'
            }

            attributions.append({
                'feature': feat,
                'label': friendly_names.get(feat, feat),
                'patient_value': round(val, 2),
                'reference_mean': round(mu, 2),
                'contribution_pct': round(risk_contribution, 2),
                'direction': 'Risk Enhancing' if risk_contribution > 0 else 'Protective',
                'abs_impact': abs(risk_contribution)
            })

        # Sort by absolute impact
        attributions.sort(key=lambda x: x['abs_impact'], reverse=True)
        return attributions[:top_n]


def generate_clinical_action_plan(
    risk_prob: float,
    ci_margin: float,
    optimal_threshold: float,
    top_factors: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates actionable clinical guidance and evidence-based next steps.
    """
    is_screen_positive = risk_prob >= optimal_threshold
    risk_pct = risk_prob * 100.0
    ci_pct = ci_margin * 100.0

    if risk_prob < 0.25:
        tier = "Low Metabolic Risk"
        badge_color = "#10b981"  # green
        recommendation = (
            "Patient demonstrates normal physiological metrics. Recommend routine biennial screening, "
            "continued adherence to standard lifestyle guidance, and regular primary care monitoring."
        )
        urgency = "Routine (12-24 Months)"
    elif risk_prob < optimal_threshold:
        tier = "Moderate / Borderline Risk"
        badge_color = "#f59e0b"  # amber
        recommendation = (
            f"Patient is in the borderline zone below the optimal clinical cutoff ({optimal_threshold*100:.1f}%). "
            "Recommend 75g Oral Glucose Tolerance Test (OGTT), HbA1c screening within 3 months, "
            "and medical nutrition therapy consultation to prevent onset."
        )
        urgency = "Medium (3-6 Months)"
    elif risk_prob < 0.70:
        tier = "Elevated Diabetes Risk (Screen Positive)"
        badge_color = "#f97316"  # orange
        recommendation = (
            f"Screening test exceeds clinical Youden cutoff ({optimal_threshold*100:.1f}%). High probability of early/latent dysglycemia. "
            "Order confirmatory venous Fasting Plasma Glucose and HbA1c. Initiate formal lifestyle modification and metabolic follow-up."
        )
        urgency = "High Priority (2-4 Weeks)"
    else:
        tier = "High-Risk / Probable Diabetes Mellitus"
        badge_color = "#ef4444"  # red
        recommendation = (
            "Strong diagnostic indication for Diabetes Mellitus. Patient exhibits severe glycemic and metabolic markers. "
            "Prompt comprehensive clinical evaluation, diagnostic laboratory workup, renal/lipid panel, and endocrinology consultation recommended."
        )
        urgency = "Urgent (< 7 Days)"

    top_drivers_text = ", ".join([f"{f['label']} ({'+' if f['contribution_pct'] > 0 else ''}{f['contribution_pct']}%)" for f in top_factors[:3]])

    return {
        'risk_tier': tier,
        'badge_color': badge_color,
        'is_screen_positive': is_screen_positive,
        'risk_percent': round(risk_pct, 1),
        'confidence_margin': round(ci_pct, 1),
        'ci_string': f"{risk_pct:.1f}% ± {ci_pct:.1f}%",
        'urgency': urgency,
        'recommendation': recommendation,
        'primary_drivers_summary': top_drivers_text
    }
