# ==============================================================================
# Chapter 5 — Customer Segmentation
# ==============================================================================
# Discovers natural customer groupings using K-Means clustering on the full
# 970K feature matrix. Segments flow downstream into Chapter 6 (churn modeling)
# as a feature and into Chapter 7 (LTV simulation) for per-segment survival
# curves and revenue profiles.
#
# Pipeline:
#   1. Load feature_matrix_modeling.parquet — no upstream rebuilding
#   2. Define clustering population (exclude ghost users only)
#   3. Test 4 feature set configurations × k=3–15 on a 200K stratified sample
#   4. Select best k and feature set based on elbow + silhouette + DB + CH
#   5. Fit final model on full 970K dataset
#   6. Profile and name each segment
#   7. Save feature_matrix_segmented.parquet
#
# K-selection decision — k=8, behavioral_plus_flags (27 features):
#   - DB minimum at k=7, CH peak at k=8, silhouette plateaus after k=10
#   - behavioral_plus_flags outperformed all other feature sets across
#     the full k range on both silhouette and DB
#   - Flags earned their inclusion: the full dataset contains structurally
#     distinct subpopulations (no-logs users, inactive payers, no-recent-
#     activity users) that the flags explicitly encode
#   - Demographics excluded — monotonically worse scores at every k tested
#
# Scaling pipeline:
#   RobustScaler was ruled out — 10 of 22 behavioral features have zero IQR
#   due to heavily modal distributions (e.g. 95% of users on 30-day plans),
#   causing divide-by-zero and scaled values up to 400+ that dominate distance
#   calculations. Final approach: winsorize at 1st/99th percentile → StandardScaler
#
# Ghost user handling:
#   Users with no transaction data are excluded from KMeans and assigned
#   segment BEST_K (integer label just outside the KMeans range) as a
#   dedicated analyst-defined segment.
#
# No-logs user handling:
#   Retained in the clustering population after a full-dataset experiment
#   confirmed they do not collapse into an artificial sentinel cluster when
#   winsorization is applied before scaling. A high-churn segment (~35%)
#   emerged from the full-data run that was not recoverable when these users
#   were excluded.
#
# Silhouette score note:
#   Scores in the 0.27–0.36 range reflect genuine behavioral complexity in
#   real-world subscription data. The 0.91 scores from an earlier clean-core
#   pipeline run were artificially inflated by sentinel vector similarity
#   (identical -1.000 feature vectors appearing perfectly clustered) and do
#   not represent meaningful cluster structure.
#
# Output: feature_matrix_segmented.parquet — 970,465 rows × 36 columns
#         (feature matrix + segment integer + segment_name string)
# ==============================================================================

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------
# Config & paths
# ------------------------------------------------------------------------------

with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = config['data']['processed_data_path']
OUTPUT_DIR     = config['data']['plots_path']

# Sampling config for k-selection
SAMPLE_SIZE     = 200_000   # stratified sample for elbow + DB + CH
SILHOUETTE_SIZE =  50_000   # smaller subset for silhouette (O(n²) cost)
K_RANGE         = range(3, 16)
RANDOM_STATE    = 42

# Final model config — determined from k-selection output
BEST_K           = 8
BEST_FEATURE_SET = 'behavioral_plus_flags'

# Segment names — determined from post-fit profiling
# Segment BEST_K is the ghost user holdout assigned outside KMeans
SEGMENT_NAMES = {
    0: 'Standard Loyal User',
    1: 'Casual Users',
    2: 'Loyal Base',
    3: 'Annual Plan High Risk',
    4: 'Inactive Payers',
    5: 'Discount Hunting Users',
    6: 'At Risk Renewers',
    7: 'Strong Long Term Users',
    8: 'Ghosts',
}

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = sns.color_palette('tab10')


# ------------------------------------------------------------------------------
# Feature set definitions
# ------------------------------------------------------------------------------

# Core behavioral features — includes log features even though no-logs users
# have -1 sentinels. Winsorization clips these to the 1st percentile of the
# full distribution before scaling, which for log features will be near zero.
BEHAVIORAL = [
    # Subscription structure
    'n_transactions_total', 'n_payment_methods', 'n_plan_days',
    'most_recent_plan_days', 'total_membership_days',
    # Payment behavior
    'total_amount_paid', 'n_cancellations', 'ever_canceled',
    'n_transactions_6m', 'n_discounted_tx', 'pct_discounted',
    'days_since_last_tx',
    # Renewal behavior
    'auto_renew_pct', 'auto_renew_current', 'auto_renew_delta',
    # Engagement (no-logs users have -1 sentinel — winsorized before scaling)
    'completion_rate', 'skip_rate', 'avg_secs_per_day',
    'days_since_last_active', 'listening_tenure_days',
    'total_days', 'total_plays',
]

FLAGS = [
    'has_no_demographics', 'has_no_logs', 'no_recent_tx_flag',
    'no_recent_log_flag', 'inactive_payer',
]

DEMOGRAPHICS = ['bd', 'gender', 'city', 'registered_via']

FEATURE_SETS = {
    'behavioral_only':              BEHAVIORAL,
    'behavioral_plus_flags':        BEHAVIORAL + FLAGS,
    'behavioral_plus_demographics': BEHAVIORAL + DEMOGRAPHICS,
    'all_features':                 BEHAVIORAL + FLAGS + DEMOGRAPHICS,
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def winsorize_features(data: pd.DataFrame, cols: list, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Clips extreme values to the 1st and 99th percentile before scaling.
    Prevents long-tail outliers from dominating KMeans distance calculations
    without distorting the bulk of the distribution.
    """
    data = data.copy()
    for col in cols:
        lo = data[col].quantile(lower)
        hi = data[col].quantile(upper)
        data[col] = data[col].clip(lo, hi)
    return data


def scale_features(df: pd.DataFrame, cols: list, scaler: StandardScaler = None):
    """
    Winsorizes then applies StandardScaler. If scaler is None, fits a new one.
    Returns (scaled array, fitted scaler) so the scaler can be reused for
    transforms on held-out data.
    """
    df_w = winsorize_features(df, cols)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_w[cols])
    else:
        X = scaler.transform(df_w[cols])
    return X, scaler


# ------------------------------------------------------------------------------
# Data loading & population split
# ------------------------------------------------------------------------------

def load_and_split():
    """
    Loads the feature matrix and splits into clustering population and ghost
    user holdout.

    Ghost users (has_no_transactions=1) are excluded from KMeans — they have
    zero payment behavior and their features are group medians of an empty
    population, providing no usable signal for any distance-based algorithm.

    No-logs users are retained — see module docstring for rationale.
    """
    df = pd.read_parquet(
        os.path.join(PROCESSED_PATH, 'feature_matrix_modeling.parquet')
    )

    ghost_mask = df['has_no_transactions'] == 1
    cluster_df = df[~ghost_mask].copy()
    ghost_df   = df[ghost_mask].copy()

    return df, cluster_df, ghost_df


# ------------------------------------------------------------------------------
# K-selection
# ------------------------------------------------------------------------------

def run_k_selection(cluster_df: pd.DataFrame) -> dict:
    """
    Evaluates 4 feature set variants across k=3–15 using four metrics:

      Inertia (elbow):          lower = better; look for rate-of-decrease knee
      Silhouette [-1,1]:        higher = better; cluster separation quality
      Davies-Bouldin [0,∞):     lower = better; intra/inter cluster ratio
      Calinski-Harabasz [0,∞):  higher = better; between/within variance ratio

    Silhouette is computed on a 50K subsample (O(n²) cost).
    All other metrics use the 200K sample.

    Scalers are fit on the 200K sample only — refitted on the full population
    in fit_final_model().

    Returns a dict keyed by feature set name containing per-k metric arrays
    and the fitted scaler for each feature set.
    """
    sample_df = resample(
        cluster_df,
        n_samples=SAMPLE_SIZE,
        stratify=cluster_df['is_churn'],
        random_state=RANDOM_STATE,
    )
    silhouette_df = resample(
        sample_df,
        n_samples=SILHOUETTE_SIZE,
        stratify=sample_df['is_churn'],
        random_state=RANDOM_STATE,
    )

    results = {}

    for fs_name, feature_cols in FEATURE_SETS.items():
        X_sample,     scaler = scale_features(sample_df,     feature_cols)
        X_silhouette, _      = scale_features(silhouette_df, feature_cols, scaler=scaler)

        inertias, sil_scores, db_scores, ch_scores = [], [], [], []
        k_values = list(K_RANGE)

        for k in k_values:
            km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=RANDOM_STATE)
            km.fit(X_sample)

            labels_sil = km.predict(X_silhouette)

            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(
                X_silhouette, labels_sil,
                sample_size=10_000, random_state=RANDOM_STATE,
            ))
            db_scores.append(davies_bouldin_score(X_silhouette, labels_sil))
            ch_scores.append(calinski_harabasz_score(X_silhouette, labels_sil))

        results[fs_name] = {
            'k_values':   k_values,
            'inertias':   inertias,
            'sil_scores': sil_scores,
            'db_scores':  db_scores,
            'ch_scores':  ch_scores,
            'scaler':     scaler,
            'features':   feature_cols,
        }

    return results


# ------------------------------------------------------------------------------
# Final model
# ------------------------------------------------------------------------------

def fit_final_model(cluster_df: pd.DataFrame, ghost_df: pd.DataFrame, df: pd.DataFrame):
    """
    Fits KMeans with BEST_K on the full clustering population using BEST_FEATURE_SET.

    Winsorization thresholds and StandardScaler are recomputed on the full
    clustering population (not reused from k-selection which fit on 200K sample).

    Ghost users are assigned segment label BEST_K — one step outside the
    KMeans integer range — as a dedicated analyst-defined segment.

    Returns df_segmented with 'segment' (int) and 'segment_name' (str) appended.
    """
    best_features = FEATURE_SETS[BEST_FEATURE_SET]

    X_full, scaler_final = scale_features(cluster_df, best_features)

    kmeans_final = KMeans(
        n_clusters=BEST_K,
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    kmeans_final.fit(X_full)

    cluster_df = cluster_df.copy()
    cluster_df['segment'] = kmeans_final.labels_

    ghost_df = ghost_df.copy()
    ghost_df['segment'] = BEST_K   # dedicated label outside KMeans range

    df_segmented = pd.concat([cluster_df, ghost_df], ignore_index=True)
    df_segmented['segment_name'] = df_segmented['segment'].map(SEGMENT_NAMES)

    return df_segmented, kmeans_final, scaler_final


# ------------------------------------------------------------------------------
# Profiling
# ------------------------------------------------------------------------------

def build_segment_profile(df_segmented: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean feature values per segment (transposed: features as rows,
    segments as columns). is_churn is appended as the final row so segment
    risk is visible alongside behavioral characteristics.
    """
    profile_cols = FEATURE_SETS[BEST_FEATURE_SET] + ['is_churn']
    return (
        df_segmented
        .groupby('segment')[profile_cols]
        .mean()
        .round(3)
        .T
    )


# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------

def plot_k_selection(results: dict):
    """Elbow + silhouette curves for all feature sets."""
    n_sets = len(results)
    fig, axes = plt.subplots(n_sets, 2, figsize=(14, 4 * n_sets))
    fig.suptitle('K-Selection: Elbow & Silhouette by Feature Set',
                 fontsize=14, fontweight='bold', y=1.01)

    for i, (fs_name, res) in enumerate(results.items()):
        k_vals = res['k_values']

        ax_elbow = axes[i, 0]
        ax_elbow.plot(k_vals, res['inertias'], marker='o', color=PALETTE[i], linewidth=2)
        ax_elbow.set_title(f'{fs_name}\nElbow Curve', fontsize=11)
        ax_elbow.set_xlabel('Number of Clusters (k)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.xaxis.set_major_locator(mticker.MultipleLocator(1))

        ax_sil = axes[i, 1]
        ax_sil.plot(k_vals, res['sil_scores'], marker='s', color=PALETTE[i + 3], linewidth=2)
        best_k = k_vals[np.argmax(res['sil_scores'])]
        ax_sil.axvline(x=best_k, color='red', linestyle='--', alpha=0.6, label=f'Best k={best_k}')
        ax_sil.set_title(f'{fs_name}\nSilhouette Score', fontsize=11)
        ax_sil.set_xlabel('Number of Clusters (k)')
        ax_sil.set_ylabel('Silhouette Score')
        ax_sil.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_sil.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'k_selection_elbow_silhouette.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_db_ch(results: dict):
    """Davies-Bouldin and Calinski-Harabasz curves for all feature sets."""
    n_sets = len(results)
    fig, axes = plt.subplots(n_sets, 2, figsize=(14, 4 * n_sets))
    fig.suptitle('K-Selection: Davies-Bouldin & Calinski-Harabasz by Feature Set',
                 fontsize=14, fontweight='bold', y=1.01)

    for i, (fs_name, res) in enumerate(results.items()):
        k_vals = res['k_values']

        # Davies-Bouldin: lower = better
        ax_db = axes[i, 0]
        ax_db.plot(k_vals, res['db_scores'], marker='o', color=PALETTE[i], linewidth=2)
        best_k_db = k_vals[np.argmin(res['db_scores'])]
        ax_db.axvline(x=best_k_db, color='red', linestyle='--', alpha=0.6, label=f'Best k={best_k_db}')
        ax_db.set_title(f'{fs_name}\nDavies-Bouldin (lower = better)', fontsize=11)
        ax_db.set_xlabel('Number of Clusters (k)')
        ax_db.set_ylabel('Davies-Bouldin Score')
        ax_db.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_db.legend(fontsize=9)

        # Calinski-Harabasz: higher = better; tends to favor higher k mechanically
        # — treat as supporting signal, not primary decision metric
        ax_ch = axes[i, 1]
        ax_ch.plot(k_vals, res['ch_scores'], marker='s', color=PALETTE[i + 3], linewidth=2)
        best_k_ch = k_vals[np.argmax(res['ch_scores'])]
        ax_ch.axvline(x=best_k_ch, color='red', linestyle='--', alpha=0.6, label=f'Best k={best_k_ch}')
        ax_ch.set_title(f'{fs_name}\nCalinski-Harabasz (higher = better)', fontsize=11)
        ax_ch.set_xlabel('Number of Clusters (k)')
        ax_ch.set_ylabel('Calinski-Harabasz Score')
        ax_ch.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_ch.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'k_selection_db_ch.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_segment_heatmap(df_segmented: pd.DataFrame):
    """
    Normalized heatmap of mean feature values per segment.
    Color = normalized to [0,1] for visual comparability across features.
    Annotation = actual mean values so absolute magnitudes are readable.
    """
    profile_cols    = FEATURE_SETS[BEST_FEATURE_SET] + ['is_churn']
    segment_profile = (
        df_segmented.groupby('segment')[profile_cols].mean().round(3).T
    )

    profile_norm = segment_profile.copy()
    for col in profile_norm.columns:
        col_min = profile_norm[col].min()
        col_max = profile_norm[col].max()
        if col_max > col_min:
            profile_norm[col] = (profile_norm[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(max(8, BEST_K * 1.5), len(profile_cols) * 0.5 + 2))
    sns.heatmap(
        profile_norm,
        annot=segment_profile,
        fmt='.2f',
        cmap='RdYlGn',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Normalized mean (0=lowest, 1=highest)'},
    )
    ax.set_title('Segment Profile Heatmap\n(color = normalized, numbers = actual mean)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'segment_profile_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_churn_by_segment(df_segmented: pd.DataFrame):
    """Bar chart of churn rate per segment with overall average reference line."""
    churn_by_seg  = df_segmented.groupby('segment')['is_churn'].mean().sort_values(ascending=False)
    overall_churn = df_segmented['is_churn'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        churn_by_seg.index.astype(str),
        churn_by_seg.values,
        color=[PALETTE[i % len(PALETTE)] for i in churn_by_seg.index],
    )
    ax.axhline(overall_churn, color='black', linestyle='--', linewidth=1.2,
               label=f'Overall churn ({overall_churn:.1%})')

    for bar, val in zip(bars, churn_by_seg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10)

    ax.set_title('Churn Rate by Segment', fontsize=12, fontweight='bold')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Churn Rate')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'churn_rate_by_segment.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_clusters(cluster_df: pd.DataFrame):
    """
    PCA projection of the 27-dimensional feature space down to 2 components.
    PC1 + PC2 capture ~44% of variance — overlap in this 2D projection does
    not imply poor separation in the full feature space.

    Visualization uses the 200K sample for speed.
    """
    SEG_COLORS_PCA = {
        0: '#4a90d9',   # Standard Loyal Users  — blue
        1: '#7ec8a0',   # Casual Users           — mint
        2: '#5ab4d4',   # Loyal Base             — teal
        3: '#e05c5c',   # Annual Plan High Risk  — red
        4: '#d9a34a',   # Inactive Payers        — amber
        5: '#9b7ed9',   # Discount Hunting Users — purple
        6: '#e07c3a',   # At-Risk Renewers       — orange
        7: '#4ad9b0',   # Strong Long-Time Users — cyan-green
    }
    SEGMENT_DISPLAY_NAMES_PCA = {
        0: 'Standard Loyal Users',
        1: 'Casual Users',
        2: 'Loyal Base',
        3: 'Annual Plan High Risk',
        4: 'Inactive Payers',
        5: 'Discount Hunting Users',
        6: 'At-Risk Renewers',
        7: 'Strong Long-Time Users',
    }

    best_features = FEATURE_SETS[BEST_FEATURE_SET]

    sample_df = resample(
        cluster_df,
        n_samples=min(200_000, len(cluster_df)),
        stratify=cluster_df['is_churn'],
        random_state=RANDOM_STATE,
    )

    X_sample, _ = scale_features(sample_df, best_features)

    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X_sample)
    km_viz = KMeans(n_clusters=BEST_K, n_init=10, random_state=RANDOM_STATE)
    labels = km_viz.fit_predict(X_sample)

    idx = np.random.choice(len(X_pca), 10_000, replace=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#161b27')
    ax.set_facecolor('#161b27')

    for seg_id in range(BEST_K):
        mask  = labels[idx] == seg_id
        color = SEG_COLORS_PCA.get(seg_id, '#888ea8')
        name  = SEGMENT_DISPLAY_NAMES_PCA.get(seg_id, f'Segment {seg_id}')
        ax.scatter(
            X_pca[idx][mask, 0], X_pca[idx][mask, 1],
            c=color, label=name,
            alpha=0.5, s=8, edgecolors='none',
        )

    ax.set_title(f'PCA Projection — k={BEST_K} Segments ({BEST_FEATURE_SET}, winsorized)',
                 fontsize=13, color='#c8d0e8', pad=12)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                  fontsize=10, color='#a0aabf')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                  fontsize=10, color='#a0aabf')
    ax.tick_params(colors='#a0aabf')
    ax.spines['bottom'].set_color('#2a2f3e')
    ax.spines['left'].set_color('#2a2f3e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#2a2f3e', linewidth=0.5, alpha=0.6)
    ax.xaxis.grid(True, color='#2a2f3e', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc='upper left',
        fontsize=9, framealpha=0.3, facecolor='#1a2035',
        labelcolor='#a0aabf', markerscale=3,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_cluster_viz.png'),
                dpi=150, bbox_inches='tight', facecolor='#161b27')
    plt.close()

    # Feature loading chart — what each PCA axis represents
    loadings = pd.DataFrame(
        pca.components_.T,
        index=best_features,
        columns=['PC1', 'PC2'],
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('PCA Feature Loadings', fontsize=13, fontweight='bold')

    for ax, pc in zip(axes, ['PC1', 'PC2']):
        sorted_loadings = loadings[pc].sort_values()
        bar_colors = ['#d62728' if v < 0 else '#1f77b4' for v in sorted_loadings]
        ax.barh(sorted_loadings.index, sorted_loadings.values, color=bar_colors)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Loading (correlation with component)', fontsize=10)
        var_pct = pca.explained_variance_ratio_[0 if pc == 'PC1' else 1] * 100
        ax.set_title(f'{pc} ({var_pct:.1f}% variance)', fontsize=11)
        ax.set_xlim(-0.6, 0.6)
        for feat in list(loadings[pc].nlargest(3).index) + list(loadings[pc].nsmallest(3).index):
            val = loadings.loc[feat, pc]
            ax.annotate(f'{val:.2f}', xy=(val, feat),
                        xytext=(5 if val > 0 else -5, 0),
                        textcoords='offset points',
                        ha='left' if val > 0 else 'right',
                        va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_loadings.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------------------------
# Save output
# ------------------------------------------------------------------------------

def save_segmented_matrix(df_segmented: pd.DataFrame):
    """Saves the feature matrix with segment columns appended."""
    out_path = os.path.join(PROCESSED_PATH, 'feature_matrix_segmented.parquet')
    df_segmented.to_parquet(out_path, index=False)


# ------------------------------------------------------------------------------
# Main — run full segmentation pipeline
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    df, cluster_df, ghost_df = load_and_split()

    # K-selection — can be skipped on re-runs since BEST_K and BEST_FEATURE_SET
    # are already set above from the original analysis
    results = run_k_selection(cluster_df)
    plot_k_selection(results)
    plot_db_ch(results)

    # Final model fit on full clustering population
    df_segmented, kmeans_final, scaler_final = fit_final_model(cluster_df, ghost_df, df)

    # Profiling and visualization
    segment_profile = build_segment_profile(df_segmented)
    plot_segment_heatmap(df_segmented)
    plot_churn_by_segment(df_segmented)
    plot_pca_clusters(cluster_df)

    save_segmented_matrix(df_segmented)