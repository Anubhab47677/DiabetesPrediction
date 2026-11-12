"""
Advanced clinical preprocessing, physiological MICE imputation, clinical biomarker engineering,
and robust feature scaling module for Diabetes Prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib

PHYSIOLOGICAL_ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
RAW_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


def engineer_advanced_clinical_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes evidence-based clinical biomarkers for metabolic disease and insulin resistance:
    1. HOMA_IR_Proxy: (Glucose * Insulin) / 405 (Standard clinical homeostasis model of insulin resistance)
    2. TyG_Index: ln(Glucose * BMI / 2) (Validated surrogate for Triglyceride-Glucose / adiposity index)
    3. Insulin_Glucose_Ratio: Insulin / (Glucose + 1e-5) (Beta-cell response metric)
    4. Age_BMI_Risk_Interaction: (Age * BMI) / 100 (Non-linear age-adiposity risk factor)
    5. Glucose_BMI_Interaction: (Glucose * BMI) / 100 (Dual glycemic-obesity burden)
    6. Metabolic_Syndrome_Score: Additive clinical count (Glucose>=100, BMI>=30, BP>=80, Age>=45)
    7. BMI_Category: WHO standard (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)
    8. Glucose_Tolerance_Category: ADA guidelines (0=Normal <100, 1=Impaired 100-125, 2=Diabetic >=126)
    9. Log_Insulin: ln(1 + Insulin) (Variance stabilization for skewed serum assays)
    """
    d = df.copy()

    # 1. HOMA-IR Proxy (Homeostatic Model Assessment of Insulin Resistance)
    # Clinical formula: (Fasting Glucose [mg/dL] * Fasting Insulin [uU/mL]) / 405
    d['HOMA_IR_Proxy'] = (d['Glucose'] * d['Insulin']) / 405.0

    # 2. TyG Surrogate Index (Triglyceride-Glucose / Body Mass surrogate)
    safe_product = np.maximum((d['Glucose'] * d['BMI']) / 2.0, 1.0)
    d['TyG_Index'] = np.log(safe_product)

    # 3. Insulin-Glucose Ratio (Reflects hyperinsulinemia relative to glycemic level)
    d['Insulin_Glucose_Ratio'] = d['Insulin'] / (d['Glucose'] + 1.0)

    # 4. Age x BMI Interaction (Adiposity effect amplifies with age)
    d['Age_BMI_Risk_Interaction'] = (d['Age'] * d['BMI']) / 100.0

    # 5. Glucose x BMI Interaction (Combined metabolic load)
    d['Glucose_BMI_Interaction'] = (d['Glucose'] * d['BMI']) / 100.0

    # 6. Composite Metabolic Syndrome Severity Score (0 to 4 flags)
    c_gluc = (d['Glucose'] >= 100.0).astype(float)
    c_bmi = (d['BMI'] >= 30.0).astype(float)
    c_bp = (d['BloodPressure'] >= 80.0).astype(float)
    c_age = (d['Age'] >= 45.0).astype(float)
    d['Metabolic_Syndrome_Score'] = c_gluc + c_bmi + c_bp + c_age

    # 7. WHO BMI Categories
    d['BMI_Category'] = pd.cut(
        d['BMI'],
        bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
        labels=[0.0, 1.0, 2.0, 3.0]
    ).astype(float)

    # 8. ADA Glucose Categories
    d['Glucose_Category'] = pd.cut(
        d['Glucose'],
        bins=[-np.inf, 99.9, 125.9, np.inf],
        labels=[0.0, 1.0, 2.0]
    ).astype(float)

    # 9. Log-transformed Insulin
    d['Log_Insulin'] = np.log1p(d['Insulin'])

    return d


class AdvancedClinicalPreprocessor:
    """
    Production-grade preprocessing pipeline featuring:
    - Biological zero replacement with NaN
    - MICE (Multivariate Imputation by Chained Equations) to condition missing values on patient covariates
    - Evidence-based clinical biomarker engineering
    - Outlier-resilient standardization via RobustScaler + StandardScaler
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # MICE Imputer: iteratively models each feature with missing values as a function of others
        self.imputer = IterativeImputer(
            max_iter=20,
            random_state=random_state,
            initial_strategy='median',
            min_value=0.0
        )
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
        self.raw_cols = RAW_FEATURES
        self.feature_names: List[str] = []
        self.is_fitted = False

    def clean_zeros(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace physiologically impossible 0s with NaN for imputation."""
        X_clean = X.copy()
        for col in PHYSIOLOGICAL_ZERO_COLS:
            if col in X_clean.columns:
                X_clean.loc[X_clean[col] == 0, col] = np.nan
        return X_clean

    def fit(self, X: pd.DataFrame):
        """Fit MICE imputer and scalers strictly on training data."""
        X_raw = X[self.raw_cols]
        X_clean = self.clean_zeros(X_raw)

        # 1. Fit MICE Imputer
        imputed_array = self.imputer.fit_transform(X_clean)
        imputed_df = pd.DataFrame(imputed_array, columns=self.raw_cols, index=X.index)

        # 2. Engineer clinical biomarkers
        engineered_df = engineer_advanced_clinical_biomarkers(imputed_df)
        self.feature_names = engineered_df.columns.tolist()

        # 3. Fit Robust Scaler then Standard Scaler
        robust_data = self.robust_scaler.fit_transform(engineered_df)
        self.standard_scaler.fit(robust_data)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new patient observations using fitted MICE imputer and scalers."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")

        X_raw = X[self.raw_cols]
        X_clean = self.clean_zeros(X_raw)

        # Impute using trained MICE model
        imputed_array = self.imputer.transform(X_clean)
        imputed_df = pd.DataFrame(imputed_array, columns=self.raw_cols, index=X.index)

        # Engineer biomarkers
        engineered_df = engineer_advanced_clinical_biomarkers(imputed_df)

        # Scale
        robust_data = self.robust_scaler.transform(engineered_df)
        scaled_data = self.standard_scaler.transform(robust_data)

        return pd.DataFrame(scaled_data, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def transform_single_patient(self, patient_dict: Dict[str, float]) -> np.ndarray:
        """Transform a single patient record from clinical form."""
        df_single = pd.DataFrame([patient_dict])
        transformed_df = self.transform(df_single)
        return transformed_df.values
