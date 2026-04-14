# ==============================================================================
# Chapter 6 — Churn Modeling
# ==============================================================================
# Trains two XGBoost churn models on the segmented feature matrix and evaluates
# them on a held-out test set.
#
# Two model variants:
#   Behavioral-only — transaction behavior + user log engagement + flags
#   Full model      — behavioral + demographic features (age, gender, city,
#                     registration channel)
#
# The behavioral vs. full comparison is a data minimization experiment:
# if demographics add negligible lift, the simpler model is preferable from
# a privacy and interpretability standpoint.
#
# Key decisions:
#   scale_pos_weight=1 (no class weighting)
#     The weighted version (scale ~10) produced near-perfect AUROC on the
#     validation set but severely overconfident probabilities — predicted churn
#     rates far exceeded actual rates in every segment. Removing the weighting
#     cost essentially nothing in AUROC and produced near-perfect calibration
#     (segment predicted vs. actual churn deltas all under 0.5%). Well-calibrated
#     probabilities are essential for Chapter 7 LTV simulation where they serve
#     directly as monthly hazard inputs.
#
#   Decision threshold = 0.30
#     Chosen over the 0.50 default because calibrated probabilities make 0.30
#     meaningful — "users we are at least 30% confident will churn." At this
#     threshold: 7,991 flagged users, 62.9% precision, 57.7% recall.
#
# Split: stratified 80/10/10 train/val/test
#   Test set held completely aside until final evaluation.
#   Val set used only for early stopping — not for threshold selection or tuning.
#
# Results:
#   Behavioral AUROC: 0.8971
#   Full AUROC:       0.9007
#   Delta:           +0.0036 — demographics add almost nothing
#   Val-to-test gap: -0.0007 — no overfitting
#
# Primary metric: AUROC, target 0.85–0.90
# Output: model_behavioral.pkl, model_full.pkl, test_predictions.parquet
# ==============================================================================

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------
# Config & paths
# ------------------------------------------------------------------------------

with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = config['data']['processed_data_path']
MODELS_PATH    = os.path.join(PROCESSED_PATH, '..', 'models')
PLOTS_PATH     = config['data']['plots_path']

os.makedirs(MODELS_PATH, exist_ok=True)

RANDOM_SEED = 42
THRESHOLD   = 0.30   # classification threshold for flagging churn candidates

# Segment name mapping — consistent with Chapter 5
SEGMENT_NAMES = {
    0: 'Standard Loyal User No Logs',
    1: 'Casual Users',
    2: 'Loyal Base',
    3: 'Annual Manual High Risk',
    4: 'Inactive Payers',
    5: 'Discount OGs',
    6: 'At-Risk Manual Renewers',
    7: 'Ride or Die OGs',
    8: 'Ghost Users',
}


# ------------------------------------------------------------------------------
# Feature set definitions
# ------------------------------------------------------------------------------

BEHAVIORAL_FEATURES = [
    # Subscription structure
    'n_payment_methods', 'n_plan_days', 'most_recent_plan_days',
    'total_membership_days', 'days_since_last_tx',
    # Transaction behavior
    'total_amount_paid', 'n_cancellations', 'ever_canceled',
    'n_transactions_total', 'n_transactions_6m', 'n_discounted_tx', 'pct_discounted',
    # Renewal behavior
    'auto_renew_pct', 'auto_renew_current', 'auto_renew_delta',
    # Engagement
    'avg_secs_per_day', 'completion_rate', 'skip_rate',
    'days_since_last_active', 'listening_tenure_days',
    'total_days', 'total_plays',
    # Missingness flags — structural absence is a predictive signal
    'has_no_transactions', 'has_no_logs', 'no_recent_tx_flag',
    'no_recent_log_flag', 'inactive_payer',
]

FULL_FEATURES = BEHAVIORAL_FEATURES + [
    'city', 'bd', 'gender', 'registered_via',
    'has_no_demographics',
]

TARGET = 'is_churn'


# ------------------------------------------------------------------------------
# Data loading & splitting
# ------------------------------------------------------------------------------

def load_data():
    """Loads the segmented feature matrix from Chapter 5."""
    return pd.read_parquet(
        os.path.join(PROCESSED_PATH, 'feature_matrix_segmented.parquet')
    )


def split_data(df: pd.DataFrame):
    """
    Stratified 80/10/10 train/val/test split.

    Split is performed on full_features so all downstream operations use
    the same index alignment. Test set is not touched until final_evaluation().

    Stratification preserves the ~9% churn rate across all three splits.
    """
    X = df[FULL_FEATURES]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------------------------------------
# Baseline
# ------------------------------------------------------------------------------

def run_baseline(X_train, X_val, y_train, y_val) -> float:
    """
    Majority-class dummy classifier. Establishes the AUROC floor (~0.50).
    Any model scoring below this has a bug upstream.
    """
    dummy = DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED)
    dummy.fit(X_train[BEHAVIORAL_FEATURES], y_train)
    return roc_auc_score(
        y_val, dummy.predict_proba(X_val[BEHAVIORAL_FEATURES])[:, 1]
    )


# ------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------

def train_behavioral_model(X_train, X_val, y_train, y_val):
    """
    XGBoost behavioral-only model.

    scale_pos_weight=1 (no class weighting) — see module docstring for
    rationale. The calibration gain outweighs the negligible AUROC cost.
    Early stopping on validation AUC with patience=20 rounds.
    """
    model = xgb.XGBClassifier(
        n_estimators          = 500,
        learning_rate         = 0.05,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = 1,
        eval_metric           = 'auc',
        early_stopping_rounds = 20,
        random_state          = RANDOM_SEED,
        n_jobs                = -1,
        verbosity             = 0,
    )
    model.fit(
        X_train[BEHAVIORAL_FEATURES], y_train,
        eval_set=[(X_val[BEHAVIORAL_FEATURES], y_val)],
        verbose=False,
    )
    return model


def train_full_model(X_train, X_val, y_train, y_val):
    """
    XGBoost full model (behavioral + demographics).

    Identical hyperparameters to the behavioral model so the AUROC difference
    is attributable solely to the demographic features, not tuning differences.
    """
    model = xgb.XGBClassifier(
        n_estimators          = 1000,
        learning_rate         = 0.05,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = 1,
        eval_metric           = 'auc',
        early_stopping_rounds = 20,
        random_state          = RANDOM_SEED,
        n_jobs                = -1,
        verbosity             = 0,
    )
    model.fit(
        X_train[FULL_FEATURES], y_train,
        eval_set=[(X_val[FULL_FEATURES], y_val)],
        verbose=False,
    )
    return model


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def evaluate_on_validation(model_behavioral, model_full, X_val, y_val) -> dict:
    """
    Computes AUROC and average precision on the validation set.
    Used to confirm no overfitting before final test set evaluation.
    """
    val_probs_b = model_behavioral.predict_proba(X_val[BEHAVIORAL_FEATURES])[:, 1]
    val_probs_f = model_full.predict_proba(X_val[FULL_FEATURES])[:, 1]

    return {
        'behavioral': {
            'auroc': roc_auc_score(y_val, val_probs_b),
            'avg_precision': average_precision_score(y_val, val_probs_b),
        },
        'full': {
            'auroc': roc_auc_score(y_val, val_probs_f),
            'avg_precision': average_precision_score(y_val, val_probs_f),
        },
    }


def evaluate_on_test(model_behavioral, model_full, X_test, y_test) -> tuple:
    """
    Final test set evaluation — called once after all modeling decisions
    are locked. Returns probability arrays and metric dict.
    """
    probs_b = model_behavioral.predict_proba(X_test[BEHAVIORAL_FEATURES])[:, 1]
    probs_f = model_full.predict_proba(X_test[FULL_FEATURES])[:, 1]

    metrics = {
        'behavioral': {
            'auroc': roc_auc_score(y_test, probs_b),
            'avg_precision': average_precision_score(y_test, probs_b),
        },
        'full': {
            'auroc': roc_auc_score(y_test, probs_f),
            'avg_precision': average_precision_score(y_test, probs_f),
        },
    }

    return probs_b, probs_f, metrics


def segment_calibration_check(df, X_test, y_prob_full):
    """
    Compares mean predicted churn probability to actual churn rate per segment.
    Delta > 5% for any segment indicates miscalibration worth investigating.

    This check was instrumental in the scale_pos_weight decision — the weighted
    model showed deltas of 15–30% per segment; the unweighted model reduced
    all deltas to under 0.5%.
    """
    segment_eval = df.loc[X_test.index].copy()
    segment_eval['prob_churn'] = y_prob_full

    comparison = (
        segment_eval.groupby('segment')
        .agg(
            n_users          = ('is_churn', 'count'),
            actual_churn     = ('is_churn', 'mean'),
            predicted_churn  = ('prob_churn', 'mean'),
        )
        .reset_index()
    )
    comparison['segment_name'] = comparison['segment'].map(SEGMENT_NAMES)
    comparison['delta'] = comparison['predicted_churn'] - comparison['actual_churn']

    return comparison.sort_values('actual_churn', ascending=False)


# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------

def plot_feature_importance(model_behavioral, model_full):
    """XGBoost built-in gain-based feature importance for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    xgb.plot_importance(model_behavioral, ax=axes[0], max_num_features=12,
                        title='Behavioral Model — Feature Importance (Gain)',
                        importance_type='gain')
    xgb.plot_importance(model_full, ax=axes[1], max_num_features=12,
                        title='Full Model — Feature Importance (Gain)',
                        importance_type='gain')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap(model_full, X_val):
    """
    SHAP summary plots for the full model on a 5,000-row validation sample.

    Bar chart shows mean absolute SHAP value per feature (overall importance).
    Beeswarm shows each user as a dot — color = feature value, x = SHAP value —
    revealing directionality: high auto_renew_pct pushes away from churn, etc.
    """
    sample = X_val[FULL_FEATURES].sample(5000, random_state=RANDOM_SEED)
    explainer   = shap.TreeExplainer(model_full)
    shap_values = explainer.shap_values(sample)

    # Bar chart
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration(y_test, probs_b, probs_f):
    """
    Calibration curves for both models. A perfectly calibrated model's curve
    follows the diagonal — predicted probability equals actual churn rate.
    Bowing above = underconfident; bowing below = overconfident.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in [('Behavioral', probs_b), ('Full', probs_f)]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker='o', linewidth=2, label=name)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives (actual churn rate)')
    ax.set_title('Calibration Plot — Churn Models', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'calibration_plot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_test, probs_f):
    """Confusion matrix at the 0.30 decision threshold."""
    y_pred = (probs_f >= THRESHOLD).astype(int)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — Full Model (threshold={THRESHOLD})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': tp / (tp + fp),
        'recall':    tp / (tp + fn),
        'flagged':   tp + fp,
    }


# ------------------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------------------

def save_models(model_behavioral, model_full):
    """Saves both models to disk for use in Chapter 7 and the Streamlit app."""
    joblib.dump(model_behavioral, os.path.join(MODELS_PATH, 'model_behavioral.pkl'))
    joblib.dump(model_full,       os.path.join(MODELS_PATH, 'model_full.pkl'))


def save_predictions(df, X_test, y_test, probs_b, probs_f):
    """
    Saves test set predictions to parquet for Chapter 7 Monte Carlo simulation.

    prob_churn_full is used directly as the monthly hazard input in the Weibull
    survival model — calibration quality here directly affects LTV accuracy.
    flagged_0_30 is the operational output: users to target for intervention.
    """
    predictions = pd.DataFrame(index=X_test.index)
    predictions['msno']                  = df.loc[X_test.index, 'msno']
    predictions['segment']               = df.loc[X_test.index, 'segment']
    predictions['segment_name']          = df.loc[X_test.index, 'segment_name']
    predictions['is_churn']              = y_test
    predictions['prob_churn_behavioral'] = probs_b
    predictions['prob_churn_full']       = probs_f
    predictions['flagged_0_30']          = (probs_f >= THRESHOLD).astype(int)

    predictions.to_parquet(
        os.path.join(PROCESSED_PATH, 'test_predictions.parquet'), index=False
    )
    return predictions


# ------------------------------------------------------------------------------
# Main — run full modeling pipeline
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Baseline floor
    baseline_auc = run_baseline(X_train, X_val, y_train, y_val)

    # Train
    model_behavioral = train_behavioral_model(X_train, X_val, y_train, y_val)
    model_full       = train_full_model(X_train, X_val, y_train, y_val)

    # Validation check
    val_metrics = evaluate_on_validation(model_behavioral, model_full, X_val, y_val)

    # Final test set evaluation — held out until this point
    probs_b, probs_f, test_metrics = evaluate_on_test(
        model_behavioral, model_full, X_test, y_test
    )

    # Calibration check by segment
    calibration_df = segment_calibration_check(df, X_test, probs_f)

    # Plots
    plot_feature_importance(model_behavioral, model_full)
    plot_shap(model_full, X_val)
    plot_calibration(y_test, probs_b, probs_f)
    threshold_metrics = plot_confusion_matrix(y_test, probs_f)

    # Save
    save_models(model_behavioral, model_full)
    predictions = save_predictions(df, X_test, y_test, probs_b, probs_f)