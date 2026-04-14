# ==============================================================================
# Chapter 3 — Feature Engineering
# ==============================================================================
# Builds the modeling feature matrix from the four processed source tables.
# All features respect the CUTOFF date (Feb 28, 2017) — no post-cutoff data
# enters the model. The cutoff parameter is threaded through every function so
# the same code can produce an EDA-mode matrix (cutoff=None) if needed later.
#
# Output: feature_matrix_modeling.parquet — 970,465 users x 34 features
#
# Feature groups:
#   Transaction features (16): behavioral signals from payment history
#   Member features (4):        demographics from registration record
#   User log features (8):      lifetime listening behavior
#   Flags (4):                  structural missingness indicators
#
# Missingness strategy:
#   - Three binary flags capture structural absence from each source table
#     (has_no_transactions, has_no_demographics, has_no_logs) — absence from a
#     table is itself a predictive signal and should not be silently imputed
#   - Within-group median fill for continuous features — prevents -1 sentinels
#     from distorting cluster distances while flags preserve the "no data" signal
#   - -1 sentinel for categorical features — treated as a distinct unknown
#     category by the model rather than collapsed with any real value
#   - 495 ghost users (missing all three sources) dropped — zero signal
#
# Key design notes:
#   - Flags are built on raw NaNs BEFORE any fillna — order is critical
#   - Natural feature range is [0, inf), so -1 is unambiguously sentinel
#   - 6-month windowed log features and trend slope deferred — requires a
#     monthly-granularity parquet pipeline (backlog item)
#   - Churn rates by coverage group documented in IMPUTATION_NOTES below
# ==============================================================================

import gc
import os
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Config & paths
# ------------------------------------------------------------------------------

with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

RAW_PATH       = config['data']['raw_data_path']
PROCESSED_PATH = config['data']['processed_data_path']

# Feature cutoff — transactions and log activity after this date are excluded
# to prevent leakage. Feb 28 2017 is the last date before the train label window.
CUTOFF    = 20170228
CUTOFF_DT = pd.to_datetime(str(CUTOFF), format='%Y%m%d')

# Column group definitions — used in imputation and downstream chapters
TX_FEATURES = [
    'n_transactions_total', 'n_payment_methods', 'n_plan_days',
    'total_amount_paid', 'n_cancellations', 'n_discounted_tx',
    'days_since_last_tx', 'ever_canceled', 'pct_discounted',
    'auto_renew_pct', 'auto_renew_current', 'most_recent_plan_days',
    'n_transactions_6m', 'no_recent_tx_flag', 'total_membership_days',
    'auto_renew_delta',
]
LOG_FEATURES = [
    'completion_rate', 'skip_rate', 'avg_secs_per_day',
    'days_since_last_active', 'listening_tenure_days',
    'no_recent_log_flag', 'total_days', 'total_plays',
]
CAT_FEATURES = ['city', 'gender', 'registered_via']


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_data():
    """
    Loads all four processed source tables from parquet.

    User logs: the pre-aggregated V1 split parquets still contain duplicate msno
    rows because a single user's source rows may have landed in multiple splits
    during the round-robin split. Re-aggregated here before feature engineering.
    The final combined parquet (user_logs_orig_agg.parquet) could also be used
    directly — both produce identical results.
    """
    members      = pd.read_parquet(os.path.join(PROCESSED_PATH, config['ingestion']['members_output']))
    train        = pd.read_parquet(os.path.join(PROCESSED_PATH, config['ingestion']['train_output']))
    transactions = pd.read_parquet(os.path.join(PROCESSED_PATH, config['ingestion']['transactions_modeling_output']))

    # Re-aggregate across splits — user rows distributed round-robin at split time
    log_splits = [
        pd.read_parquet(os.path.join(PROCESSED_PATH, f'user_logs_v1_agg_{i}.parquet'))
        for i in range(1, 6)
    ]
    user_logs = pd.concat(log_splits, ignore_index=True)
    del log_splits
    gc.collect()

    user_logs = user_logs.groupby('msno').agg(
        total_days         = ('total_days',         'sum'),
        total_secs         = ('total_secs',         'sum'),
        total_plays        = ('total_plays',        'sum'),
        total_unique_songs = ('total_unique_songs', 'sum'),
        num_100_sum        = ('num_100_sum',        'sum'),
        num_25_sum         = ('num_25_sum',         'sum'),
        last_active_date   = ('last_active_date',   'max'),
        first_active_date  = ('first_active_date',  'min')
    ).reset_index()

    return members, train, transactions, user_logs


# ------------------------------------------------------------------------------
# Transaction feature engineering
# ------------------------------------------------------------------------------

def build_transaction_features(transactions: pd.DataFrame, cutoff: int = None) -> pd.DataFrame:
    """
    Builds per-user transaction features from the KKBox transactions table.

    All features use data through cutoff (default: max date in table).

    6-month window logic:
      n_transactions_6m counts non-cancel transactions in [cutoff-6m, cutoff].
      -1 means the user had no transactions in the window (absent from window),
      which is distinct from 0 (present but no activity). The no_recent_tx_flag
      makes this distinction explicit for the model.

    Auto-renew delta:
      Negative delta = user recently disabled auto-renew relative to their
      historical average. Captures behavioral shift without fixed-window
      sparsity issues on users with long histories.

    Membership tenure:
      Computed via interval merging to avoid double-counting plan changes and
      same-day records. Returns true paid days only.
    """
    df = transactions.copy()

    df['transaction_date']       = pd.to_datetime(df['transaction_date'],       format='%Y%m%d')
    df['membership_expire_date'] = pd.to_datetime(df['membership_expire_date'], format='%Y%m%d')

    cutoff_dt = pd.to_datetime(str(cutoff), format='%Y%m%d') if cutoff else df['transaction_date'].max()
    df = df[df['transaction_date'] <= cutoff_dt].copy()

    window_6m_start = cutoff_dt - pd.DateOffset(months=6)

    df['discount']      = df['plan_list_price'] - df['actual_amount_paid']
    df['is_discounted'] = (df['discount'] > 0).astype(int)

    # Exclude cancellations from auto-renew features — cancellations carry
    # is_auto_renew=0 by definition, which would suppress the genuine renewal
    # preference signal if included
    non_cancel = df[df['is_cancel'] == 0].copy()

    # Core aggregations across full history
    agg = df.groupby('msno').agg(
        n_transactions_total = ('transaction_date',   'count'),
        n_payment_methods    = ('payment_method_id',  'nunique'),
        n_plan_days          = ('payment_plan_days',  'nunique'),
        total_amount_paid    = ('actual_amount_paid', 'sum'),
        n_cancellations      = ('is_cancel',          'sum'),
        n_discounted_tx      = ('is_discounted',      'sum'),
        days_since_last_tx   = ('transaction_date',   lambda x: (cutoff_dt - x.max()).days),
    ).reset_index()

    agg['ever_canceled']  = (agg['n_cancellations'] > 0).astype(int)
    agg['pct_discounted'] = agg['n_discounted_tx'] / agg['n_transactions_total']

    # Historical auto-renew rate (non-cancel only)
    auto_renew_pct = (
        non_cancel.groupby('msno')['is_auto_renew']
        .mean()
        .rename('auto_renew_pct')
        .reset_index()
    )

    # Most recent non-cancel transaction — current plan state
    auto_renew_current = (
        non_cancel.sort_values('transaction_date')
        .groupby('msno')
        .tail(1)[['msno', 'is_auto_renew', 'payment_plan_days']]
        .rename(columns={
            'is_auto_renew':     'auto_renew_current',
            'payment_plan_days': 'most_recent_plan_days',
        })
    )

    # 6-month window transaction count
    # Left join from all users — NaN after join means absent from window → -1
    all_users = df[['msno']].drop_duplicates()
    nc_6m     = non_cancel[non_cancel['transaction_date'] >= window_6m_start]
    n_tx_6m   = (
        nc_6m.groupby('msno')
        .size()
        .rename('n_transactions_6m')
        .reset_index()
    )
    n_tx_6m = all_users.merge(n_tx_6m, on='msno', how='left')
    n_tx_6m['no_recent_tx_flag'] = n_tx_6m['n_transactions_6m'].isna().astype(int)
    n_tx_6m['n_transactions_6m'] = n_tx_6m['n_transactions_6m'].fillna(-1).astype(int)

    # Membership tenure via interval merging — avoids double-counting overlapping
    # subscription periods caused by plan changes and same-day records
    def compute_tenure(group):
        intervals = sorted(zip(group['transaction_date'], group['membership_expire_date']))
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return sum(max((e - s).days, 0) for s, e in merged)

    tenure = (
        df.groupby('msno')
        .apply(compute_tenure)
        .rename('total_membership_days')
        .reset_index()
    )

    features = (
        agg
        .merge(auto_renew_pct,     on='msno', how='left')
        .merge(auto_renew_current, on='msno', how='left')
        .merge(n_tx_6m,            on='msno', how='left')
        .merge(tenure,             on='msno', how='left')
    )

    # Auto-renew delta — negative = recently disabled relative to historical avg
    features['auto_renew_delta'] = features['auto_renew_current'] - features['auto_renew_pct']
    features['auto_renew_pct']   = features['auto_renew_pct'].fillna(0.0)

    return features


# ------------------------------------------------------------------------------
# Member feature engineering
# ------------------------------------------------------------------------------

def build_member_features(members: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts demographic features from the members table.

    bd (birth year proxy):
      Values outside [8, 75] replaced with -2 (invalid/implausible age).
      -1 is reserved for structurally missing users (not in members table),
      so a distinct sentinel (-2) is needed for present-but-invalid records.

    gender encoding:
      male=1, female=0, unknown/missing=2
      2 is distinct from -1 (missing demographics entirely) so the model
      learns separate behavior for users who chose not to disclose vs. users
      with no registration record at all.

    reg_year excluded — total_membership_days (from transactions) captures
    tenure more precisely and without the ceiling effect near the cutoff.
    """
    df = members.copy()

    df['bd']     = df['bd'].where(df['bd'].between(8, 75), other=-2)
    df['gender'] = df['gender'].map({'male': 1, 'female': 0}).fillna(2).astype(int)

    return df[['msno', 'city', 'bd', 'gender', 'registered_via']].copy()


# ------------------------------------------------------------------------------
# User log feature engineering
# ------------------------------------------------------------------------------

def build_user_log_features(user_logs: pd.DataFrame, cutoff_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Builds per-user behavioral features from pre-aggregated user log parquets.

    All features are lifetime (full history through cutoff).
    6-month windowed counts and trend slope are deferred pending a
    monthly-granularity parquet pipeline (backlog item).

    Rates are computed here rather than at ingestion to keep the ingestion
    layer free of business logic.

    no_recent_log_flag mirrors the 6-month window logic in transaction features
    so both recency signals use a consistent definition of "recent".

    -1 sentinel for users absent from logs entirely is handled downstream
    during feature matrix imputation — not set here.
    """
    df = user_logs.copy()

    df['last_active_date']  = pd.to_datetime(df['last_active_date'])
    df['first_active_date'] = pd.to_datetime(df['first_active_date'])

    # Replace 0 plays/days with NaN to avoid division artifacts — zero-play
    # edge cases are filled to 0.0 at the end of this function
    total_plays = df['total_plays'].replace(0, np.nan)
    total_days  = df['total_days'].replace(0, np.nan)

    df['completion_rate']  = df['num_100_sum'] / total_plays
    df['skip_rate']        = df['num_25_sum']  / total_plays
    df['avg_secs_per_day'] = df['total_secs']  / total_days

    df['days_since_last_active'] = (cutoff_dt - df['last_active_date']).dt.days
    df['listening_tenure_days']  = (df['last_active_date'] - df['first_active_date']).dt.days

    # Consistent 6m recency window with transaction features
    window_6m_start = cutoff_dt - pd.DateOffset(months=6)
    df['no_recent_log_flag'] = (df['last_active_date'] < window_6m_start).astype(int)

    features = df[[
        'msno',
        'completion_rate',
        'skip_rate',
        'avg_secs_per_day',
        'days_since_last_active',
        'listening_tenure_days',
        'no_recent_log_flag',
        'total_days',
        'total_plays',
    ]].copy()

    # Zero-play edge cases produce NaN rates — fill to 0.0 (no activity recorded)
    features[['completion_rate', 'skip_rate', 'avg_secs_per_day']] = (
        features[['completion_rate', 'skip_rate', 'avg_secs_per_day']].fillna(0.0)
    )

    return features


# ------------------------------------------------------------------------------
# Feature matrix assembly
# ------------------------------------------------------------------------------

def assemble_feature_matrix(train, transaction_features, members_features, user_log_features):
    """
    Left-joins all feature groups onto the train label set.

    Left join on train preserves all 970,960 labeled users regardless of
    source table coverage. NaNs after joining indicate structural absence
    from that source — flags are built on these raw NaNs before any fillna.
    """
    feature_matrix = (
        train[['msno', 'is_churn']]
        .merge(transaction_features, on='msno', how='left')
        .merge(members_features,     on='msno', how='left')
        .merge(user_log_features,    on='msno', how='left')
    )
    return feature_matrix


# ------------------------------------------------------------------------------
# Missingness flags
# ------------------------------------------------------------------------------

def add_missingness_flags(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Adds four binary indicator columns capturing structural data gaps.

    MUST be called on the raw post-join DataFrame before any fillna —
    flags are derived from NaN presence and will be incorrect if imputation
    runs first.

    has_no_transactions / has_no_demographics / has_no_logs:
      Absence from a source table is itself a predictive signal. These flags
      allow the model to learn distinct behavior for each coverage group rather
      than treating missing values as a random nuisance.

    inactive_payer:
      Paying users who have lifetime log data but have gone silent in the last
      6 months. Lower churn rate than average — likely long-tenure subscribers
      who pay reliably but listen infrequently. Definition relies on
      no_recent_log_flag as a proxy for the 6m window; revisit when windowed
      log counts are available.
    """
    fm = feature_matrix.copy()

    fm['has_no_transactions'] = fm['n_payment_methods'].isna().astype(int)
    fm['has_no_demographics'] = fm['city'].isna().astype(int)
    fm['has_no_logs']         = fm['avg_secs_per_day'].isna().astype(int)

    fm['inactive_payer'] = (
        fm['n_payment_methods'].notna() &   # has transaction history
        (fm['ever_canceled'] == 0) &        # never canceled
        (fm['has_no_logs'] == 0) &          # has lifetime log data
        (fm['no_recent_log_flag'] == 1)     # silent in last 6 months
    ).astype(int)

    return fm


# ------------------------------------------------------------------------------
# Ghost user removal
# ------------------------------------------------------------------------------

def drop_ghost_users(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Removes 495 users with no data across all three source tables.

    Ghost users have zero predictive signal — all features are imputed from
    group medians, making their model scores unreliable. Their 47.3% churn
    rate is noise given the small sample size, not a learnable pattern.
    """
    ghost_mask = (
        (feature_matrix['has_no_transactions'] == 1) &
        (feature_matrix['has_no_demographics'] == 1) &
        (feature_matrix['has_no_logs'] == 1)
    )
    return feature_matrix[~ghost_mask].reset_index(drop=True)


# ------------------------------------------------------------------------------
# Imputation
# ------------------------------------------------------------------------------

def impute_feature_matrix(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Applies two-strategy imputation after flags are set and ghost users dropped.

    Strategy 1 — within-group median fill (continuous features):
      Medians computed separately per missingness group so a no-log user is
      filled from other no-log users, not from active listeners. Prevents
      -1 sentinels from distorting cluster distances in segmentation.

    Strategy 2 — -1 sentinel (categorical features):
      city, gender, registered_via treated as discrete unknown category.
      Distinct from gender=2 (user actively chose not to disclose) and
      bd=-2 (present but invalid age value).

    Known imputation outcomes by coverage group:
      Full data (tx + mem + logs)    805,465  9.4% churn   core active base
      Missing logs only               53,473  8.1%         active payers, light listeners
      Missing mem only                    22  54.5%        tiny group, high churn
      Missing tx only                    252  72.2%        never paid — lapsing
      tx only (missing mem + logs)   109,476  5.1%         loyal anonymous payers
      mem only (missing tx + logs)     1,777  54.2%        registered, never engaged
    """
    fm = feature_matrix.copy()

    no_tx   = fm['has_no_transactions'] == 1
    no_logs = fm['has_no_logs'] == 1
    no_mem  = fm['has_no_demographics'] == 1

    # Within-group median fill — transactions
    tx_medians = fm.loc[no_tx, TX_FEATURES].median()
    fm.loc[no_tx, TX_FEATURES] = fm.loc[no_tx, TX_FEATURES].fillna(tx_medians)

    # Within-group median fill — user logs
    log_medians = fm.loc[no_logs, LOG_FEATURES].median()
    fm.loc[no_logs, LOG_FEATURES] = fm.loc[no_logs, LOG_FEATURES].fillna(log_medians)

    # bd — within-group median fill (continuous demographic)
    bd_median = fm.loc[no_mem, 'bd'].median()
    fm.loc[no_mem, 'bd'] = fm.loc[no_mem, 'bd'].fillna(bd_median)

    # Categorical demographics — -1 as distinct unknown category
    fm[CAT_FEATURES] = fm[CAT_FEATURES].fillna(-1).astype(int)

    return fm


# ------------------------------------------------------------------------------
# Post-imputation validation
# ------------------------------------------------------------------------------

def validate_feature_matrix(feature_matrix: pd.DataFrame) -> bool:
    """
    Validates feature ranges and flag integrity after imputation.
    Returns True if all checks pass, False otherwise.

    avg_secs_per_day: clipped to 86,400 (seconds in a day) if any overflow
    artifacts survived from the source data — this is applied in-place.
    """
    fm = feature_matrix

    # Range checks — natural lower bound is 0 (or -1/-2 for sentinels)
    checks = {
        'avg_secs_per_day':      (-1,   86_400),
        'total_membership_days': (-1,    5_000),
        'days_since_last_tx':    (-1,    2_000),
        'total_amount_paid':     (-1,  500_000),
        'completion_rate':       (-1,        1),
        'skip_rate':             (-1,        1),
        'auto_renew_pct':        (-1,        1),
        'n_cancellations':       (-1,      100),
        'bd':                    (-2,       75),
    }

    all_passed = True
    for col, (expected_min, expected_max) in checks.items():
        actual_min = fm[col].min()
        actual_max = fm[col].max()
        if not (actual_min >= expected_min and actual_max <= expected_max):
            all_passed = False

    # Clip avg_secs_per_day at physical ceiling — overflow artifacts in source
    feature_matrix['avg_secs_per_day'] = feature_matrix['avg_secs_per_day'].clip(upper=86_400)

    # No NaNs should remain after imputation
    if feature_matrix.isnull().sum().sum() > 0:
        all_passed = False

    return all_passed


# ------------------------------------------------------------------------------
# Segmentation imputation
# (same strategy, called separately before Chapter 5 clustering)
# ------------------------------------------------------------------------------

def apply_segmentation_imputation(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Alias for impute_feature_matrix — exists for clarity in the segmentation
    chapter where imputation is re-applied after the feature matrix is loaded.
    Identical logic; kept separate to make the chapter dependency explicit.
    """
    return impute_feature_matrix(feature_matrix)


# ------------------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------------------

def save_features(transaction_features, members_features, user_log_features, feature_matrix):
    """
    Saves individual feature tables and the final modeling matrix to parquet.
    feature_matrix_modeling.parquet is the primary input to all downstream chapters.
    """
    saves = {
        'transaction_features_modeling.parquet': transaction_features,
        'members_features_modeling.parquet':     members_features,
        'user_log_features_modeling.parquet':    user_log_features,
        'feature_matrix_modeling.parquet':       feature_matrix,
    }
    for filename, df in saves.items():
        path = os.path.join(PROCESSED_PATH, filename)
        df.to_parquet(path, index=False)


# ------------------------------------------------------------------------------
# Main — run full feature engineering pipeline
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    members, train, transactions, user_logs = load_data()

    transaction_features = build_transaction_features(transactions, cutoff=CUTOFF)
    members_features     = build_member_features(members)
    user_log_features    = build_user_log_features(user_logs, CUTOFF_DT)

    feature_matrix = assemble_feature_matrix(
        train, transaction_features, members_features, user_log_features
    )
    feature_matrix = add_missingness_flags(feature_matrix)
    feature_matrix = drop_ghost_users(feature_matrix)
    feature_matrix = impute_feature_matrix(feature_matrix)

    validate_feature_matrix(feature_matrix)

    save_features(transaction_features, members_features, user_log_features, feature_matrix)