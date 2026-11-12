"""
Baseline machine learning models training, benchmarking, and evaluation.
Trains and compares Logistic Regression, KNN, SVM, Random Forest, and Gradient Boosting.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)


def get_baseline_models() -> Dict[str, Any]:
    """Instantiate the 5 baseline machine learning classifiers with fixed seeds."""
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }


def evaluate_classifier(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Computes all standard classification metrics with emphasis on medical recall.
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(specificity),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        },
        'pr_curve': {
            'precision': prec_curve.tolist(),
            'recall': rec_curve.tolist()
        }
    }


def train_and_evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Trains all baseline models, evaluates performance, and returns trained models,
    detailed evaluation metrics, and a comparison summary DataFrame.
    """
    models = get_baseline_models()
    trained_models = {}
    detailed_metrics = {}
    summary_rows = []

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        metrics = evaluate_classifier(model, X_test.values, y_test.values)
        metrics['training_time_sec'] = round(training_time, 4)

        trained_models[name] = model
        detailed_metrics[name] = metrics

        summary_rows.append({
            'Model': name,
            'Accuracy': round(metrics['accuracy'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall (Sensitivity)': round(metrics['recall'], 4),
            'F1-Score': round(metrics['f1'], 4),
            'ROC-AUC': round(metrics['roc_auc'], 4),
            'False Negatives (FN)': metrics['fn'],
            'False Positives (FP)': metrics['fp'],
            'Training Time (s)': metrics['training_time_sec']
        })

    summary_df = pd.DataFrame(summary_rows)
    return trained_models, detailed_metrics, summary_df
