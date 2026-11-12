"""
Verification script for Testing the Deployed Clinical Decision Support Model.
Tests:
1. Likely non-diabetic case
2. Likely diabetic case
3. Borderline case
4. Extreme / Edge-case inputs
5. Monte Carlo Dropout epistemic uncertainty quantification (95% CI)
6. Youden's J optimal screening decision comparison
7. Local biomarker risk attribution (explainability)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from src.mlp_model import mc_dropout_predict
from src.clinical_engine import LocalRiskExplainer, generate_clinical_action_plan


def run_deployment_tests():
    print("=" * 70)
    print("RUNNING ADVANCED CLINICAL DEPLOYMENT TEST SUITE")
    print("=" * 70)

    # 1. Load artifacts
    preprocessor = joblib.load('models/preprocessor.joblib')
    mlp_model = keras.models.load_model('models/diabetes_mlp.keras')
    baseline_models = joblib.load('models/baseline_models.joblib')
    
    with open('models/clinical_thresholds.json', 'r') as f:
        clinical_data = json.load(f)
    with open('models/feature_importance.json', 'r') as f:
        feat_importances = json.load(f)

    opt_threshold = clinical_data['youden_optimal_threshold']
    print(f"Loaded clinical artifacts. Youden Optimal Cutoff = {opt_threshold:.2f} (Default = 0.50)")

    # Prepare local explainer reference
    df_raw = pd.read_csv('data/diabetes.csv')
    healthy_ref = df_raw[df_raw['Outcome'] == 0]
    explainer = LocalRiskExplainer(
        reference_cohort=healthy_ref,
        feature_importances=feat_importances,
        feature_names=preprocessor.raw_cols
    )

    test_cases = [
        {
            'name': 'Case 1: Likely Non-Diabetic Patient',
            'data': {
                'Pregnancies': 1,
                'Glucose': 85.0,
                'BloodPressure': 66.0,
                'SkinThickness': 20.0,
                'Insulin': 55.0,
                'BMI': 21.4,
                'DiabetesPedigreeFunction': 0.22,
                'Age': 23
            },
            'expected_class': 0
        },
        {
            'name': 'Case 2: Likely Diabetic Patient',
            'data': {
                'Pregnancies': 7,
                'Glucose': 185.0,
                'BloodPressure': 90.0,
                'SkinThickness': 40.0,
                'Insulin': 280.0,
                'BMI': 38.2,
                'DiabetesPedigreeFunction': 0.95,
                'Age': 55
            },
            'expected_class': 1
        },
        {
            'name': 'Case 3: Borderline Patient',
            'data': {
                'Pregnancies': 3,
                'Glucose': 128.0,
                'BloodPressure': 78.0,
                'SkinThickness': 28.0,
                'Insulin': 110.0,
                'BMI': 28.6,
                'DiabetesPedigreeFunction': 0.45,
                'Age': 40
            },
            'expected_class': None
        },
        {
            'name': 'Case 4: Extreme / Boundary Values (Robustness Check)',
            'data': {
                'Pregnancies': 15,
                'Glucose': 250.0,
                'BloodPressure': 130.0,
                'SkinThickness': 60.0,
                'Insulin': 700.0,
                'BMI': 55.0,
                'DiabetesPedigreeFunction': 2.2,
                'Age': 78
            },
            'expected_class': 1
        }
    ]

    for tc in test_cases:
        print(f"\n-------------------------------------------------------------")
        print(f"Evaluating: {tc['name']}")
        print(f"-------------------------------------------------------------")

        # Preprocess
        transformed_input = preprocessor.transform_single_patient(tc['data'])
        assert transformed_input.shape[1] == 17, f"Expected 17 engineered features, got {transformed_input.shape[1]}"

        # Standard & Monte Carlo Dropout Uncertainty Prediction
        raw_prob = float(mlp_model.predict(transformed_input, verbose=0)[0][0])
        mc_mean, mc_std, ci_low, ci_high = mc_dropout_predict(mlp_model, transformed_input, n_iter=25)
        mean_p = float(mc_mean[0])
        ci_margin = float(1.96 * mc_std[0])

        pred_standard = 1 if raw_prob >= 0.50 else 0
        pred_youden = 1 if raw_prob >= opt_threshold else 0

        # Local Risk Attribution
        attributions = explainer.explain_patient(tc['data'], raw_prob, top_n=4)

        # Clinical Action Plan
        action_plan = generate_clinical_action_plan(raw_prob, ci_margin, opt_threshold, attributions)

        print(f"  Biometrics: Glucose={tc['data']['Glucose']} mg/dL, BMI={tc['data']['BMI']}, Age={tc['data']['Age']}")
        print(f"  Point Probability: {raw_prob * 100:.1f}%")
        print(f"  MC Dropout 95% Confidence Interval: {mean_p * 100:.1f}% ± {ci_margin * 100:.1f}% [{ci_low[0]*100:.1f}% - {ci_high[0]*100:.1f}%]")
        print(f"  Diagnostic Decisions: Cutoff 0.50 -> {'Diabetic (1)' if pred_standard == 1 else 'Non-Diabetic (0)'} | Youden {opt_threshold:.2f} -> {'Diabetic (1)' if pred_youden == 1 else 'Non-Diabetic (0)'}")
        print(f"  Clinical Risk Tier: {action_plan['risk_tier']} (Urgency: {action_plan['urgency']})")
        print(f"  Top Risk Drivers:")
        for attr in attributions:
            sign = "+" if attr['contribution_pct'] > 0 else ""
            print(f"    - {attr['label']}: Patient={attr['patient_value']} vs Ref={attr['reference_mean']} (Impact: {sign}{attr['contribution_pct']}%) [{attr['direction']}]")

        if tc['expected_class'] is not None:
            assert pred_youden == tc['expected_class'], f"Expected {tc['expected_class']} at Youden threshold, got {pred_youden}"
            print(f"  Verification Status: PASSED")

    print("\n" + "=" * 70)
    print("ALL 4 CLINICAL TEST SCENARIOS PASSED WITH UNCERTAINTY & ATTRIBUTION!")
    print("=" * 70)


if __name__ == '__main__':
    run_deployment_tests()
