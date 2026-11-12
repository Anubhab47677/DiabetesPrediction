"""
Clinical Deep Residual Multilayer Perceptron (Res-MLP) architecture,
cost-sensitive training, Monte Carlo Dropout uncertainty quantification,
and hyperparameter tuning module for Diabetes Prediction.
Built with TensorFlow / Keras.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.utils.class_weight import compute_class_weight


def set_reproducible_seed(seed: int = 42):
    """Ensure consistent weight initialization and training reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_residual_mlp_model(
    input_dim: int,
    dropout_rate: float = 0.20,
    l2_reg: float = 0.0002,
    learning_rate: float = 0.001,
    optimizer_name: str = 'adam'
) -> keras.Model:
    """
    Constructs a Clinical Deep Residual Multilayer Perceptron (Res-MLP).
    Engineered for tabular clinical genomics/metabolomics:
      Input (input_dim)
      -> Dense(64, Swish) + BatchNorm + Dropout(drop)
      -> Residual Block:
           x_sub = Dense(64, Swish) -> BatchNorm -> Dropout(drop*0.75) -> Dense(64, Swish) -> BatchNorm
           Skip Connection: Add(x0, x_sub)
      -> Dense(32, Swish) + BatchNorm + Dropout(drop*0.5)
      -> Dense(1, Sigmoid)
    """
    inputs = layers.Input(shape=(input_dim,), name="clinical_features_input")

    # Initial Feature Transformation & Dense Representation
    x0 = layers.Dense(
        64,
        activation='swish',
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="dense_in_64"
    )(inputs)
    x0 = layers.BatchNormalization(name="bn_in")(x0)
    x0 = layers.Dropout(dropout_rate, name="drop_in")(x0)

    # Residual Block with Skip Connection
    res_sub = layers.Dense(
        64,
        activation='swish',
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="res_dense_1"
    )(x0)
    res_sub = layers.BatchNormalization(name="res_bn_1")(res_sub)
    res_sub = layers.Dropout(dropout_rate * 0.75, name="res_drop_1")(res_sub)

    res_sub = layers.Dense(
        64,
        activation='swish',
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="res_dense_2"
    )(res_sub)
    res_sub = layers.BatchNormalization(name="res_bn_2")(res_sub)

    # Residual Addition (Skip Connection)
    res_out = layers.Add(name="residual_skip_connection")([x0, res_sub])

    # Dense Decision Layer
    x = layers.Dense(
        32,
        activation='swish',
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="dense_decision_32"
    )(res_out)
    x = layers.BatchNormalization(name="bn_decision")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_decision")(x)

    # Sigmoid Output
    outputs = layers.Dense(1, activation='sigmoid', name="output_risk_prob")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Clinical_Res_MLP_Classifier")

    if optimizer_name.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model


def build_mlp_model(
    input_dim: int,
    hidden_layers_units: List[int] = [64, 32],
    activation: str = 'relu',
    dropout_rate: float = 0.2,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001,
    optimizer_name: str = 'adam'
) -> keras.Model:
    """
    Standard sequential Multilayer Perceptron for baseline comparisons and ablations.
    """
    model = keras.Sequential(name="Diabetes_MLP_Classifier")
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_layers_units):
        model.add(layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f"dense_layer_{i+1}"
        ))
        model.add(layers.BatchNormalization(name=f"batch_norm_{i+1}"))
        if dropout_rate > 0.0:
            model.add(layers.Dropout(rate=dropout_rate, name=f"dropout_{i+1}"))

    model.add(layers.Dense(1, activation='sigmoid', name="output_layer_sigmoid"))

    if optimizer_name.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model


def get_balanced_class_weights(y_train: np.ndarray, positive_penalty_factor: float = 1.0) -> Dict[int, float]:
    """
    Computes class weights inversely proportional to class frequencies,
    with an optional clinical penalty factor to penalize False Negatives.
    """
    classes = np.unique(y_train)
    raw_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, raw_weights)}
    if 1 in class_weight_dict and positive_penalty_factor != 1.0:
        class_weight_dict[1] = class_weight_dict[1] * positive_penalty_factor
    return class_weight_dict


def train_mlp_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 120,
    batch_size: int = 32,
    patience: int = 18,
    class_weight: Optional[Dict[int, float]] = None
) -> Tuple[keras.Model, Dict[str, List[float]], float]:
    """
    Trains the MLP model with Early Stopping, dynamic learning rate decay,
    and optional cost-sensitive class weights.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-5,
            verbose=0
        )
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0
    )
    training_time = time.time() - start_time
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}

    return model, history_dict, round(training_time, 4)


def evaluate_mlp_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates the model on unseen test data, computing accuracy, precision,
    recall, specificity, F1, ROC-AUC, Brier score, and curves.
    """
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_test, y_prob)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    brier = brier_score_loss(y_test, y_prob)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(specificity),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'brier_score': float(brier),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'roc_curve': {
            'fpr': [float(x) for x in fpr],
            'tpr': [float(x) for x in tpr],
            'thresholds': [float(x) for x in roc_thresh]
        },
        'pr_curve': {
            'precision': [float(x) for x in prec_curve],
            'recall': [float(x) for x in rec_curve]
        },
        'probabilities': [float(x) for x in y_prob]
    }


def mc_dropout_predict(
    model: keras.Model,
    X: np.ndarray,
    n_iter: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout Inference for epistemic uncertainty estimation.
    Executes n_iter stochastic forward passes with dropout active (training=True).
    Returns:
      (mean_prob, std_dev, ci_lower, ci_upper)
    """
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    predictions = []
    for _ in range(n_iter):
        preds = model(X_tensor, training=True)
        predictions.append(preds.numpy().ravel())

    predictions = np.array(predictions)  # shape: (n_iter, n_samples)
    mean_prob = np.mean(predictions, axis=0)
    std_dev = np.std(predictions, axis=0)
    ci_lower = np.clip(mean_prob - 1.96 * std_dev, 0.0, 1.0)
    ci_upper = np.clip(mean_prob + 1.96 * std_dev, 0.0, 1.0)

    return mean_prob, std_dev, ci_lower, ci_upper


def perform_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Systematic tuning comparing standard architectures against the Clinical Res-MLP.
    """
    set_reproducible_seed(42)
    input_dim = X_train.shape[1]
    class_weights = get_balanced_class_weights(y_train, positive_penalty_factor=1.15)

    trial_configs = [
        {'type': 'standard', 'hidden_layers': [32], 'lr': 0.001, 'batch_size': 32, 'dropout': 0.0, 'opt': 'adam', 'desc': 'Shallow (1 Layer, No Dropout)'},
        {'type': 'standard', 'hidden_layers': [64, 32], 'lr': 0.001, 'batch_size': 32, 'dropout': 0.2, 'opt': 'adam', 'desc': 'Standard 2-Layer (Moderate Dropout)'},
        {'type': 'standard', 'hidden_layers': [128, 64, 32], 'lr': 0.001, 'batch_size': 32, 'dropout': 0.3, 'opt': 'adam', 'desc': 'Deep 3-Layer Architecture'},
        {'type': 'standard', 'hidden_layers': [64, 32], 'lr': 0.001, 'batch_size': 32, 'dropout': 0.2, 'opt': 'rmsprop', 'desc': '2-Layer with RMSprop'},
        {'type': 'residual', 'lr': 0.001, 'batch_size': 32, 'dropout': 0.20, 'opt': 'adam', 'desc': 'Clinical Res-MLP (Dense 64 + ResBlock 64 + Dense 32, Swish)'},
        {'type': 'residual', 'lr': 0.0008, 'batch_size': 16, 'dropout': 0.15, 'opt': 'adam', 'desc': 'Clinical Res-MLP (Low LR + Small Batch)'}
    ]

    results = []
    best_config = None
    best_composite_score = -1.0
    best_trained_model = None
    best_history = None

    for i, cfg in enumerate(trial_configs):
        if cfg.get('type') == 'residual':
            model = build_residual_mlp_model(
                input_dim=input_dim,
                dropout_rate=cfg['dropout'],
                learning_rate=cfg['lr'],
                optimizer_name=cfg['opt']
            )
        else:
            model = build_mlp_model(
                input_dim=input_dim,
                hidden_layers_units=cfg['hidden_layers'],
                dropout_rate=cfg['dropout'],
                learning_rate=cfg['lr'],
                optimizer_name=cfg['opt']
            )

        trained_m, history, t_time = train_mlp_model(
            model, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=cfg['batch_size'], patience=15,
            class_weight=class_weights
        )

        val_eval = evaluate_mlp_model(trained_m, X_val, y_val)
        val_auc = val_eval['roc_auc']
        val_acc = val_eval['accuracy']
        val_rec = val_eval['recall']
        val_f1 = val_eval['f1']

        results.append({
            'Trial': i + 1,
            'Description': cfg['desc'],
            'Architecture': 'Res-MLP' if cfg.get('type') == 'residual' else 'Sequential',
            'Dropout': cfg['dropout'],
            'Learning Rate': cfg['lr'],
            'Batch Size': cfg['batch_size'],
            'Optimizer': cfg['opt'].upper(),
            'Val Accuracy': round(val_acc, 4),
            'Val Recall': round(val_rec, 4),
            'Val F1-Score': round(val_f1, 4),
            'Val ROC-AUC': round(val_auc, 4),
            'Training Time (s)': t_time
        })

        composite_score = 0.55 * val_auc + 0.45 * val_rec
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_config = cfg
            best_trained_model = trained_m
            best_history = history

    tuning_df = pd.DataFrame(results)
    return {
        'best_config': best_config,
        'best_model': best_trained_model,
        'best_history': best_history,
        'tuning_table': tuning_df
    }
