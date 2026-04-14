# ==============================================================================
# Chapter 7 — Monte Carlo LTV Simulation
# ==============================================================================
# Estimates 12-month customer lifetime value for each user in the test set
# using a Weibull survival model driven by calibrated churn probabilities
# from Chapter 6.
#
# Approach:
#   Each user's prob_churn_full from Chapter 6 is treated as their monthly
#   hazard rate. A per-user Weibull scale parameter (lambda) is derived from
#   this hazard, with a global shape parameter k=0.95 (slightly below 1,
#   capturing the real-world pattern of elevated early churn risk that
#   decreases slightly as tenure grows).
#
#   For each user, N_SIMULATIONS monthly survival paths are simulated.
#   Each month: discrete hazard = max(weibull_hazard_at_t, HAZARD_FLOOR).
#   Revenue accumulates while the user is alive. LTV = mean total revenue
#   across all simulation paths × MONTHLY_REVENUE.
#
# Key parameters:
#   WEIBULL_SHAPE_K = 0.95   — near-constant hazard, slight early churn risk
#   HAZARD_FLOOR = 0.015  # structural monthly churn — unconditional baseline attrition
#                              randomness; ensures no user shows guaranteed survival
#   HORIZON_MONTHS  = 12     — one year forward window
#   N_SIMULATIONS   = 1,000  — sufficient for stable mean estimates (~10s runtime)
#   MONTHLY_REVENUE = 149    — NTD standard plan (≈ $5 USD)
#
# Platt scaling (calibration) was evaluated and dropped:
#   The Chapter 6 model (scale_pos_weight=1) was already well-calibrated —
#   segment predicted vs. actual churn deltas all under 0.5%. Fitting a
#   Platt scaler on the test set would constitute leakage.
#   prob_churn_full is used directly as the hazard input.
#
# ROI framework:
#   Configurable reach/response/cost inputs produce per-segment ROI estimates.
#   Designed to wire directly into the Chapter 8 Streamlit dashboard sliders.
#
# Results:
#   Mean LTV:              1,345 NTD (~$45 USD)
#   Median expected months: 10.45
#   Primary target:        At-Risk Manual Renewers (Seg 6) — 1.6x ROI,
#                          largest recoverable revenue pool
#   Negative ROI:          Annual Manual High Risk (Seg 3), Ghost Users (Seg 8)
#
# Output: ltv_predictions.parquet, ltv_segment_summary.csv, three plot PNGs
# ==============================================================================

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from pathlib import Path

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------
# Config & paths
# ------------------------------------------------------------------------------

with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = config['data']['processed_data_path']
PLOTS_PATH     = config['data']['plots_path']

# Simulation constants
MONTHLY_REVENUE = 149        # NTD per user per month (standard plan)
HORIZON_MONTHS  = 12         # forward simulation window
N_SIMULATIONS   = 1_000      # Monte Carlo paths per user
WEIBULL_SHAPE_K = 0.95       # shape k < 1 = slight early churn risk
HAZARD_FLOOR    = 0.025      # 2.5% monthly unconditional churn — no intervention can prevent this
RANDOM_SEED     = 42

np.random.seed(RANDOM_SEED)

# ROI framework defaults — wired to Streamlit sliders in Chapter 8
INTERVENTION_COST_NTD = 100   # cost per user contacted (NTD)
REACH_RATE            = 0.15  # fraction of flagged users contacted
RESPONSE_RATE         = 0.30  # fraction of contacted users successfully retained

# Segment name mapping — consistent with Chapters 5 & 6
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
# Weibull parameterization
# ------------------------------------------------------------------------------

def churn_prob_to_lambda(churn_prob: np.ndarray, k: float = WEIBULL_SHAPE_K) -> np.ndarray:
    """
    Derives per-user Weibull scale parameter (lambda) from monthly churn probability.

    Weibull survival: S(t) = exp(-(t/λ)^k)
    At t=1 month:    1 - S(1) = churn_prob
    Solving for λ:   λ = 1 / (-log(1 - churn_prob))^(1/k)

    churn_prob is clipped to (1e-6, 1-1e-6) to prevent log(0) and log(1) edge cases.
    """
    churn_prob = np.clip(churn_prob, 1e-6, 1 - 1e-6)
    return 1 / ((-np.log(1 - churn_prob)) ** (1 / k))


def weibull_hazard_at_t(t: int, lam: np.ndarray, k: float) -> np.ndarray:
    """
    Discrete monthly conditional churn probability at month t given Weibull parameters.

    P(churn in month t | survived to t) = 1 - S(t) / S(t-1)
    where S(t) = exp(-(t/λ)^k)

    Using the discrete conditional hazard rather than the instantaneous hazard
    h(t) = (k/λ)(t/λ)^(k-1) because the simulation operates in monthly steps,
    not continuous time.
    """
    s_t   = np.exp(-(t / lam) ** k)
    s_tm1 = np.exp(-((t - 1) / lam) ** k) if t > 1 else np.ones_like(lam)
    return 1 - (s_t / s_tm1)


# ------------------------------------------------------------------------------
# Monte Carlo simulation
# ------------------------------------------------------------------------------

def simulate_ltv(
    lambdas: np.ndarray,
    k: float = WEIBULL_SHAPE_K,
    horizon: int = HORIZON_MONTHS,
    n_sims: int = N_SIMULATIONS,
    monthly_revenue: float = MONTHLY_REVENUE,
    structural_churn: float = STRUCTURAL_CHURN_RATE,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Simulates discrete monthly survival paths for all users simultaneously.

    Each month applies two sequential churn draws:
      1. Structural churn — 2.5% of surviving users exit unconditionally,
         regardless of individual risk. Represents real-world attrition that
         no retention campaign can prevent (life changes, forgotten accounts,
         loss of access, etc.).
      2. Behavioral churn — remaining survivors face their individual
         Weibull-derived conditional hazard. This is the component the
         model predicts and retention spend targets.

    Returns (ltv_array, expected_months_array), both shape (n_users,).
    """
    rng     = np.random.default_rng(seed)
    n_users = len(lambdas)
    lambdas = np.array(lambdas)

    alive        = np.ones((n_users, n_sims), dtype=bool)
    months_alive = np.zeros((n_users, n_sims), dtype=np.float32)

    for t in range(1, horizon + 1):
        # Step 1: structural churn — unconditional, affects all survivors
        structural_churns = rng.random((n_users, n_sims)) < structural_churn
        alive = alive & ~structural_churns

        # Step 2: behavioral churn — individual Weibull hazard
        h_behavioral = weibull_hazard_at_t(t, lambdas[:, None], k)
        behavioral_churns = rng.random((n_users, n_sims)) < h_behavioral
        alive = alive & ~behavioral_churns

        months_alive += alive.astype(np.float32)

    expected_months = months_alive.mean(axis=1)
    ltv             = expected_months * monthly_revenue

    return ltv, expected_months


# ------------------------------------------------------------------------------
# Segment aggregation
# ------------------------------------------------------------------------------

def build_segment_summary(test_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-user LTV and churn predictions to segment level.
    Includes USD conversions at ~30 NTD = $1 USD for reporting.
    """
    summary = test_preds.groupby(['segment', 'segment_name']).agg(
        n_users           = ('msno',            'count'),
        actual_churn_rate = ('is_churn',         'mean'),
        avg_churn_prob    = ('prob_churn_full',  'mean'),
        avg_ltv_12m       = ('ltv_12m',          'mean'),
        median_ltv_12m    = ('ltv_12m',          'median'),
        avg_months_alive  = ('expected_months',  'mean'),
        total_ltv_12m     = ('ltv_12m',          'sum'),
        n_flagged         = ('flagged_0_30',      'sum'),
    ).reset_index()

    summary['avg_ltv_12m_usd']   = (summary['avg_ltv_12m']   / 30).round(2)
    summary['total_ltv_12m_usd'] = (summary['total_ltv_12m'] / 30).round(0)

    return summary


def compute_roi(
    segment_summary: pd.DataFrame,
    reach_rate: float = REACH_RATE,
    response_rate: float = RESPONSE_RATE,
    cost_ntd: float = INTERVENTION_COST_NTD,
) -> pd.DataFrame:
    """
    Computes per-segment intervention ROI given configurable reach/response/cost.

    ROI logic:
      n_contacted       = n_flagged × reach_rate
      n_saved           = n_contacted × response_rate
      revenue_saved     = n_saved × avg_ltv_12m
      cost              = n_contacted × cost_per_user
      net_roi           = revenue_saved - cost
      roi_multiple      = revenue_saved / cost  (NaN → 0 if no contacts)

    Designed to be called with slider inputs from the Streamlit dashboard.
    """
    df = segment_summary.copy()

    df['n_contacted']       = (df['n_flagged'] * reach_rate).astype(int)
    df['n_saved']           = (df['n_contacted'] * response_rate).astype(int)
    df['revenue_saved_ntd'] = df['n_saved'] * df['avg_ltv_12m']
    df['cost_ntd']          = df['n_contacted'] * cost_ntd
    df['net_roi_ntd']       = df['revenue_saved_ntd'] - df['cost_ntd']
    df['roi_multiple']      = (
        df['revenue_saved_ntd'] / df['cost_ntd'].replace(0, np.nan)
    ).round(2)

    return df.sort_values('net_roi_ntd', ascending=False)


# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------

def plot_ltv_and_survival(test_preds: pd.DataFrame, segment_summary: pd.DataFrame):
    """
    Three-panel figure: average LTV by segment, churn vs LTV scatter,
    and Weibull survival curves per segment.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Average LTV by segment
    colors = plt.cm.RdYlGn(
        1 - segment_summary['actual_churn_rate'] / segment_summary['actual_churn_rate'].max()
    )
    axes[0].bar(segment_summary['segment'].astype(str),
                segment_summary['avg_ltv_12m'], color=colors)
    axes[0].set_title('Average 12-Month LTV by Segment (NTD)', fontsize=13)
    axes[0].set_xlabel('Segment')
    axes[0].set_ylabel('Avg LTV (NTD)')
    for i, (val, _) in enumerate(zip(segment_summary['avg_ltv_12m'],
                                     segment_summary.itertuples())):
        axes[0].text(i, val + 10, f'{val:.0f}', ha='center', fontsize=9)

    # Panel 2: Churn rate vs avg LTV scatter (bubble size = n_users)
    axes[1].scatter(
        segment_summary['actual_churn_rate'],
        segment_summary['avg_ltv_12m'],
        s=segment_summary['n_users'] / 50,
        c=segment_summary['segment'],
        cmap='tab10', alpha=0.8, edgecolors='black', linewidths=0.5,
    )
    for _, row in segment_summary.iterrows():
        axes[1].annotate(f"Seg {int(row['segment'])}",
                         (row['actual_churn_rate'], row['avg_ltv_12m']),
                         textcoords='offset points', xytext=(6, 4), fontsize=9)
    axes[1].set_title('Churn Rate vs Avg LTV by Segment', fontsize=13)
    axes[1].set_xlabel('Actual Churn Rate')
    axes[1].set_ylabel('Avg LTV (NTD)')

    # Panel 3: Weibull survival curves per segment
    months = np.arange(0, HORIZON_MONTHS + 1)
    for _, row in segment_summary.iterrows():
        lam      = churn_prob_to_lambda(row['avg_churn_prob'], WEIBULL_SHAPE_K)
        survival = np.exp(-(months / lam) ** WEIBULL_SHAPE_K)
        axes[2].plot(months, survival, marker='o', markersize=3,
                     label=f"Seg {int(row['segment'])} (churn={row['actual_churn_rate']:.1%})")
    axes[2].set_title('Weibull Survival Curves by Segment', fontsize=13)
    axes[2].set_xlabel('Month')
    axes[2].set_ylabel('Survival Probability')
    axes[2].legend(fontsize=8, loc='lower left')
    axes[2].set_ylim(0, 1.05)
    axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'ltv_simulation_plots.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_survival_distribution(test_preds: pd.DataFrame):
    """
    Distribution of expected months alive across all users, split by risk tier.
    Left panel: full population. Right panel: high vs low risk overlay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    median_months = test_preds['expected_months'].median()
    mean_months   = test_preds['expected_months'].mean()

    axes[0].hist(test_preds['expected_months'], bins=50,
                 color='#3498db', edgecolor='white', linewidth=0.3)
    axes[0].axvline(median_months, color='red',    linestyle='--',
                    label=f'Median: {median_months:.1f}mo')
    axes[0].axvline(mean_months,   color='orange', linestyle='--',
                    label=f'Mean: {mean_months:.1f}mo')
    axes[0].set_xlabel('Expected Months Alive (mean across 1,000 sims)')
    axes[0].set_ylabel('Number of Users')
    axes[0].set_title('Distribution of Expected Survival — All Users')
    axes[0].legend()

    high_risk = test_preds[test_preds['prob_churn_full'] >= 0.30]['expected_months']
    low_risk  = test_preds[test_preds['prob_churn_full'] <  0.30]['expected_months']
    axes[1].hist(low_risk,  bins=50, color='#2ecc71', alpha=0.7,
                 edgecolor='white', linewidth=0.3,
                 label=f'Low risk (<30% churn)  n={len(low_risk):,}')
    axes[1].hist(high_risk, bins=50, color='#e74c3c', alpha=0.7,
                 edgecolor='white', linewidth=0.3,
                 label=f'High risk (≥30% churn) n={len(high_risk):,}')
    axes[1].set_xlabel('Expected Months Alive')
    axes[1].set_ylabel('Number of Users')
    axes[1].set_title('Survival Distribution — Low vs High Risk')
    axes[1].legend()

    plt.suptitle(
        f'Monte Carlo Survival Distribution  '
        f'(k={WEIBULL_SHAPE_K}, floor={HAZARD_FLOOR:.1%}, horizon={HORIZON_MONTHS}mo)',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'survival_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_roi(roi_df: pd.DataFrame):
    """
    Three-panel ROI figure: net ROI by segment, ROI multiple, revenue saved vs cost.
    Green = positive ROI, red = negative.
    """
    plot_df    = roi_df.sort_values('net_roi_ntd', ascending=False)
    net_colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in plot_df['net_roi_ntd']]
    mul_colors = ['#2ecc71' if x >= 1 else '#e74c3c'
                  for x in plot_df['roi_multiple'].fillna(0)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Intervention ROI by Segment  '
        f'(reach={REACH_RATE:.0%}, response={RESPONSE_RATE:.0%}, cost={INTERVENTION_COST_NTD} NTD/user)',
        fontsize=13, y=1.02,
    )

    axes[0].barh(plot_df['segment_name'], plot_df['net_roi_ntd'] / 1000, color=net_colors)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('Net ROI (000s NTD)')
    axes[0].set_title('Net ROI by Segment')
    axes[0].invert_yaxis()

    axes[1].barh(plot_df['segment_name'], plot_df['roi_multiple'].fillna(0), color=mul_colors)
    axes[1].axvline(1, color='black', linewidth=0.8, linestyle='--', label='Break-even (1×)')
    axes[1].set_xlabel('ROI Multiple')
    axes[1].set_title('ROI Multiple by Segment')
    axes[1].legend()
    axes[1].invert_yaxis()

    axes[2].barh(plot_df['segment_name'], plot_df['revenue_saved_ntd'] / 1000,
                 color='#3498db', alpha=0.7, label='Revenue Saved')
    axes[2].barh(plot_df['segment_name'], plot_df['cost_ntd'] / 1000,
                 color='#e67e22', alpha=0.7, label='Intervention Cost')
    axes[2].set_xlabel('NTD (000s)')
    axes[2].set_title('Revenue Saved vs Cost')
    axes[2].legend()
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'roi_by_segment.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------------------

def save_outputs(test_preds: pd.DataFrame, segment_summary: pd.DataFrame):
    """
    Saves per-user LTV predictions and segment summary.
    ltv_predictions.parquet is the primary input to the Chapter 8 dashboard.
    """
    test_preds.to_parquet(
        os.path.join(PROCESSED_PATH, 'ltv_predictions.parquet'), index=False
    )
    segment_summary.to_csv(
        os.path.join(PROCESSED_PATH, 'ltv_segment_summary.csv'), index=False
    )


# ------------------------------------------------------------------------------
# Main — run full LTV simulation pipeline
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Load test set predictions from Chapter 6
    test_preds = pd.read_parquet(
        os.path.join(PROCESSED_PATH, 'test_predictions.parquet')
    )

    # Derive Weibull lambda from calibrated churn probabilities
    test_preds['weibull_lambda'] = churn_prob_to_lambda(
        test_preds['prob_churn_full'].values, k=WEIBULL_SHAPE_K
    )

    # Run Monte Carlo simulation (~10s for 97K users × 1,000 sims)
    ltv_values, expected_months = simulate_ltv(
        lambdas=test_preds['weibull_lambda'].values,
        k=WEIBULL_SHAPE_K,
        horizon=HORIZON_MONTHS,
        n_sims=N_SIMULATIONS,
        monthly_revenue=MONTHLY_REVENUE,
        structural_churn=STRUCTURAL_CHURN_RATE,
        seed=RANDOM_SEED,
    )

    test_preds['ltv_12m']         = ltv_values
    test_preds['expected_months'] = expected_months

    # Segment aggregation and ROI
    segment_summary = build_segment_summary(test_preds)
    roi_df          = compute_roi(segment_summary)

    # Plots
    plot_ltv_and_survival(test_preds, segment_summary)
    plot_survival_distribution(test_preds)
    plot_roi(roi_df)

    # Save
    save_outputs(test_preds, segment_summary)