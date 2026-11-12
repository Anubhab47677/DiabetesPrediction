"""
Data loader and dataset inspection module for Diabetes Prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# Clinical features where 0 is physiologically implausible
PHYSIOLOGICAL_ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

FEATURE_DESCRIPTIONS = {
    'Pregnancies': 'Number of times pregnant',
    'Glucose': 'Plasma glucose concentration (2 hours in an oral glucose tolerance test, mg/dL)',
    'BloodPressure': 'Diastolic blood pressure (mm Hg)',
    'SkinThickness': 'Triceps skin fold thickness (mm)',
    'Insulin': '2-Hour serum insulin (mu U/ml)',
    'BMI': 'Body mass index (weight in kg / (height in m)^2)',
    'DiabetesPedigreeFunction': 'Diabetes pedigree function (genetic risk score based on family history)',
    'Age': 'Age in years',
    'Outcome': 'Target class (0 = Non-Diabetic, 1 = Diabetic)'
}

NORMAL_REFERENCE_RANGES = {
    'Glucose': '70 - 99 mg/dL (fasting), < 140 mg/dL (post-prandial)',
    'BloodPressure': '60 - 80 mm Hg (diastolic)',
    'SkinThickness': '10 - 50 mm',
    'Insulin': '16 - 166 mu U/ml (post-prandial)',
    'BMI': '18.5 - 24.9 kg/m^2',
    'DiabetesPedigreeFunction': '0.08 - 2.42 (statistical index)',
    'Age': '21 - 81 years'
}


def load_raw_data(filepath: str = 'data/diabetes.csv') -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def inspect_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect physiologically implausible zero values in clinical features.
    Returns a DataFrame with count and percentage of zero values per column.
    """
    stats = []
    total_records = len(df)
    for col in PHYSIOLOGICAL_ZERO_COLS:
        zero_count = int((df[col] == 0).sum())
        pct = float((zero_count / total_records) * 100)
        stats.append({
            'Feature': col,
            'Zero Count': zero_count,
            'Zero Percentage (%)': round(pct, 2),
            'Physiological Implausibility': 'Cannot be 0 in living humans; indicates missing test record'
        })
    return pd.DataFrame(stats)


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Provide high-level metadata about the dataset."""
    return {
        'total_samples': len(df),
        'total_features': df.shape[1] - 1,
        'feature_names': [col for col in df.columns if col != 'Outcome'],
        'target_variable': 'Outcome',
        'non_diabetic_count': int((df['Outcome'] == 0).sum()),
        'diabetic_count': int((df['Outcome'] == 1).sum()),
        'class_distribution': {
            'Non-Diabetic (0)': f"{(df['Outcome'] == 0).mean() * 100:.2f}%",
            'Diabetic (1)': f"{(df['Outcome'] == 1).mean() * 100:.2f}%"
        },
        'missing_values_explicit': int(df.isnull().sum().sum()),
        'zero_anomalies_summary': inspect_zero_values(df).to_dict(orient='records')
    }
