# ==============================================================================
# Chapter 2 — Data Ingestion
# ==============================================================================
# Responsibility: load all four raw KKBox source tables, cache to parquet,
# and verify cross-table coverage. No business logic, no feature decisions,
# and no cutoff filtering live here — this is the single source of truth for
# raw → processed conversion.
#
# Source tables:
#   train                — churn labels; users with membership expiring Feb 2017
#   members              — demographics: age, gender, city, registration date
#   transactions (v1+v2) — conscious user actions only; auto-renewals are silent
#   user_logs (v1+v2)    — daily listening behavior
#
# User logs large-file strategy:
#   user_logs.csv (~160M rows, ~30 GB) exceeds available RAM and cannot be
#   loaded directly. Pipeline:
#     1. Split into 5 files (~40M rows each) via PowerShell StreamReader/
#        StreamWriter — reads line-by-line with near-zero RAM usage
#     2. Process each split independently in pandas with chunked aggregation
#     3. Save per-split parquets, then combine into user_logs_orig_agg.parquet
#     4. user_logs_v2.csv (March 2017, ~1.4 GB) is small enough to process
#        directly without splitting
#
# Two bugs fixed vs. original implementation:
#   - Integer overflow: total_secs cast to float32 before summing — raw int64
#     values overflow when accumulated across a user's full multi-year history
#   - Mean-of-means error: avg_secs_per_day and avg_plays_per_day were
#     previously computed as the mean of chunk-level means, which is incorrect
#     when chunks have unequal user row counts. Both are now deferred to the
#     combine step and computed as total / total_days on the fully combined data
#
# Transactions split:
#   transactions_modeling.parquet — filtered to <= Feb 28, 2017 (feature cutoff)
#   transactions_eda.parquet      — full history through Mar 2017
# ==============================================================================

import gc
import os
import subprocess
import time

import numpy as np
import pandas as pd
import yaml


# ------------------------------------------------------------------------------
# Config & paths
# ------------------------------------------------------------------------------

with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

RAW_PATH       = config['data']['raw_data_path']
PROCESSED_PATH = config['data']['processed_data_path']
CHUNK_SIZE     = config['ingestion']['chunk_size']
N_SPLITS       = config['ingestion']['n_splits']

os.makedirs(PROCESSED_PATH, exist_ok=True)

# V1 split paths — files are pre-split on disk (see split_user_logs_v1)
V1_SPLITS = [os.path.join(RAW_PATH, f"user_logs_split_{i+1}.csv") for i in range(N_SPLITS)]

# V2 — single file, small enough to process without splitting
V2_PATH = os.path.join(RAW_PATH, config['files']['user_logs_v2'])


# ------------------------------------------------------------------------------
# Helper: chunk-level aggregation (shared by V1 splits and V2)
# ------------------------------------------------------------------------------

def _agg_log_chunks(filepath, chunk_size):
    """
    Reads a user_logs CSV in chunks and returns a per-user aggregated DataFrame.

    Aggregation is two-level:
      Level 1 (chunk): group by msno within each chunk
      Level 2 (file):  sum/min/max across all chunk-level aggregates

    Derived rate metrics (avg_secs_per_day etc.) are intentionally NOT computed
    here — they must be calculated after all splits are combined to avoid the
    mean-of-means error.
    """
    agg_chunks = []

    for chunk in pd.read_csv(
            filepath, chunksize=chunk_size,
            dtype={'msno': 'str', 'date': 'int32', 'num_25': 'int32',
                   'num_50': 'int32', 'num_75': 'int32', 'num_985': 'int32',
                   'num_100': 'int32', 'num_unq': 'int32', 'total_secs': 'float32'}):

        chunk['date'] = pd.to_datetime(chunk['date'].astype(str), format='%Y%m%d')
        chunk['num_plays'] = (
            chunk['num_25'] + chunk['num_50'] + chunk['num_75'] +
            chunk['num_985'] + chunk['num_100']
        )
        # Clip to physically valid range — values > 86400 sec/day are source artifacts
        chunk['total_secs'] = chunk['total_secs'].clip(lower=0, upper=86_400)

        chunk_agg = chunk.groupby('msno').agg(
            total_days         = ('date',       'count'),
            total_secs         = ('total_secs', 'sum'),
            total_plays        = ('num_plays',  'sum'),
            total_unique_songs = ('num_unq',    'sum'),
            num_100_sum        = ('num_100',    'sum'),
            num_25_sum         = ('num_25',     'sum'),
            last_active_date   = ('date',       'max'),
            first_active_date  = ('date',       'min')
        ).reset_index()

        agg_chunks.append(chunk_agg)

    combined = pd.concat(agg_chunks, ignore_index=True)

    return combined.groupby('msno').agg(
        total_days         = ('total_days',         'sum'),
        total_secs         = ('total_secs',         'sum'),
        total_plays        = ('total_plays',        'sum'),
        total_unique_songs = ('total_unique_songs', 'sum'),
        num_100_sum        = ('num_100_sum',        'sum'),
        num_25_sum         = ('num_25_sum',         'sum'),
        last_active_date   = ('last_active_date',   'max'),
        first_active_date  = ('first_active_date',  'min')
    ).reset_index()


# ------------------------------------------------------------------------------
# Step 1 — Split user_logs.csv (V1) via PowerShell
# ------------------------------------------------------------------------------

def split_user_logs_v1():
    """
    Splits the ~30 GB user_logs.csv into N_SPLITS files using PowerShell
    StreamReader/StreamWriter — reads one line at a time with near-zero RAM usage.

    Rows are distributed round-robin so each split contains a representative
    sample of users rather than a chronological slice. Because a single user's
    rows may land across multiple files, Cell 5 re-aggregates after combining.

    Idempotent — skipped if all split files already exist on disk.
    """
    if all(os.path.exists(f) for f in V1_SPLITS):
        return

    user_logs_v1_path = os.path.join(RAW_PATH, config['files']['user_logs_v1'])

    ps_script = f"""
    $reader = [System.IO.StreamReader]::new('{user_logs_v1_path}')
    $header = $reader.ReadLine()
    $writers = @()
    for ($i = 1; $i -le {N_SPLITS}; $i++) {{
        $path = '{RAW_PATH}\\user_logs_split_' + $i + '.csv'
        $writers += [System.IO.StreamWriter]::new($path)
        $writers[-1].WriteLine($header)
    }}
    $lineCount = 0
    while (($line = $reader.ReadLine()) -ne $null) {{
        $writers[$lineCount % {N_SPLITS}].WriteLine($line)
        $lineCount++
    }}
    $reader.Close()
    foreach ($w in $writers) {{ $w.Close() }}
    Write-Host "Done. Total lines: $lineCount"
    """

    result = subprocess.run(['powershell', '-Command', ps_script], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"PowerShell split failed:\n{result.stderr}")


# ------------------------------------------------------------------------------
# Step 2 — Aggregate V1 splits
# ------------------------------------------------------------------------------

def aggregate_v1_splits():
    """
    Processes each of the 5 V1 split files independently via chunked aggregation
    and saves a per-split parquet. Idempotent — each split is skipped if its
    parquet already exists.
    """
    for split_idx, split_file in enumerate(V1_SPLITS, start=1):
        out_path = os.path.join(PROCESSED_PATH, f"user_logs_v1_agg_{split_idx}.parquet")

        if os.path.exists(out_path):
            continue

        split_df = _agg_log_chunks(split_file, CHUNK_SIZE)
        split_df.to_parquet(out_path, index=False)
        del split_df
        gc.collect()


# ------------------------------------------------------------------------------
# Step 3 — Aggregate user_logs_v2.csv directly
# ------------------------------------------------------------------------------

def aggregate_v2():
    """
    Aggregates user_logs_v2.csv (March 2017, ~1.4 GB) using the same schema as
    the V1 splits. Idempotent — skipped if parquet already exists.
    """
    out_path = os.path.join(PROCESSED_PATH, "user_logs_v2_agg.parquet")

    if os.path.exists(out_path):
        return

    v2_agg = _agg_log_chunks(V2_PATH, CHUNK_SIZE)
    v2_agg.to_parquet(out_path, index=False)
    del v2_agg
    gc.collect()


# ------------------------------------------------------------------------------
# Step 4 — Combine V1 splits + V2 into final user logs parquet
# ------------------------------------------------------------------------------

def combine_user_logs():
    """
    Concatenates all 5 V1 split parquets and the V2 parquet, then re-aggregates
    by msno so each user has exactly one row covering their full history.

    Re-aggregation is necessary because the round-robin split means a user's
    rows are distributed across multiple split files.

    Derived rate metrics are computed here — after combining — to avoid the
    mean-of-means error that would result from computing them per-split.

    Idempotent — skipped if final parquet already exists.
    """
    out_path = os.path.join(PROCESSED_PATH, config['ingestion']['user_logs_output'])

    if os.path.exists(out_path):
        return

    v1_parts = [
        pd.read_parquet(os.path.join(PROCESSED_PATH, f"user_logs_v1_agg_{i}.parquet"))
        for i in range(1, N_SPLITS + 1)
    ]
    v2_agg = pd.read_parquet(os.path.join(PROCESSED_PATH, "user_logs_v2_agg.parquet"))

    combined = pd.concat(v1_parts + [v2_agg], ignore_index=True)
    del v1_parts, v2_agg
    gc.collect()

    # Ensure datetime types survived the parquet round-trip before aggregating
    combined['last_active_date']  = pd.to_datetime(combined['last_active_date'],  errors='coerce')
    combined['first_active_date'] = pd.to_datetime(combined['first_active_date'], errors='coerce')

    user_logs = combined.groupby('msno').agg(
        total_days         = ('total_days',         'sum'),
        total_secs         = ('total_secs',         'sum'),
        total_plays        = ('total_plays',        'sum'),
        total_unique_songs = ('total_unique_songs', 'sum'),
        num_100_sum        = ('num_100_sum',        'sum'),
        num_25_sum         = ('num_25_sum',         'sum'),
        last_active_date   = ('last_active_date',   'max'),
        first_active_date  = ('first_active_date',  'min')
    ).reset_index()

    del combined
    gc.collect()

    # Derived metrics — computed once on fully combined data
    user_logs['avg_secs_per_day']  = user_logs['total_secs']  / user_logs['total_days']
    user_logs['avg_plays_per_day'] = user_logs['total_plays'] / user_logs['total_days']

    # completion_rate: fraction of plays listened to completion
    # skip_rate: fraction of plays abandoned in first 25%
    # Users with zero total_plays receive 0 for both rates
    user_logs['completion_rate'] = (
        user_logs['num_100_sum'] / user_logs['total_plays']
    ).replace([np.inf, np.nan], 0)

    user_logs['skip_rate'] = (
        user_logs['num_25_sum'] / user_logs['total_plays']
    ).replace([np.inf, np.nan], 0)

    user_logs.to_parquet(out_path, index=False)
    del user_logs
    gc.collect()


# ------------------------------------------------------------------------------
# Step 5 — Build transactions parquets (modeling + EDA splits)
# ------------------------------------------------------------------------------

def build_transactions():
    """
    Combines transactions_v1 and transactions_v2, deduplicates, then saves two
    versioned parquets:

      transactions_modeling.parquet — filtered to <= Feb 28, 2017
                                      (respects the feature cutoff for modeling)
      transactions_eda.parquet      — full history through Mar 2017
                                      (used for EDA and LTV simulation)

    The original v1/v2 parquets are deleted after both outputs are confirmed on
    disk to avoid ambiguity about which file to use downstream.

    Idempotent — skipped if both output parquets already exist.
    """
    modeling_path = os.path.join(PROCESSED_PATH, config['ingestion']['transactions_modeling_output'])
    eda_path      = os.path.join(PROCESSED_PATH, config['ingestion']['transactions_eda_output'])

    if os.path.exists(modeling_path) and os.path.exists(eda_path):
        return

    tx_v1 = pd.read_parquet(os.path.join(PROCESSED_PATH, 'transactions_v1.parquet'))
    tx_v2 = pd.read_parquet(os.path.join(PROCESSED_PATH, 'transactions_v2.parquet'))

    combined = pd.concat([tx_v1, tx_v2], ignore_index=True)
    del tx_v1, tx_v2
    gc.collect()

    combined = combined.drop_duplicates().reset_index(drop=True)
    combined['transaction_date'] = pd.to_datetime(
        combined['transaction_date'].astype(str), format='%Y%m%d', errors='coerce'
    )

    # EDA version — full history
    combined.to_parquet(eda_path, index=False)

    # Modeling version — cutoff at Feb 28, 2017 to prevent feature leakage
    tx_modeling = combined[combined['transaction_date'] <= '2017-02-28'].copy().reset_index(drop=True)
    tx_modeling.to_parquet(modeling_path, index=False)

    del combined, tx_modeling
    gc.collect()

    # Remove superseded raw parquets — modeling and eda versions are canonical
    for filename in ['transactions_v1.parquet', 'transactions_v2.parquet']:
        old_path = os.path.join(PROCESSED_PATH, filename)
        if os.path.exists(old_path):
            os.remove(old_path)


# ------------------------------------------------------------------------------
# Step 6 — Cross-table coverage check
# ------------------------------------------------------------------------------

def check_coverage():
    """
    Returns a dict summarising how many train users appear in each supporting
    table. Gaps here become missing-value flags in feature engineering
    (has_no_transactions, has_no_logs, has_no_demographics).

    Expected results:
      members        ~98%
      transactions   ~84%  (auto-renewals don't generate transaction records)
      user_logs      ~96%
      all three      ~81%
    """
    train     = pd.read_parquet(os.path.join(PROCESSED_PATH, config['ingestion']['train_output']))
    members   = pd.read_parquet(os.path.join(PROCESSED_PATH, config['ingestion']['members_output']))
    tx        = pd.read_parquet(os.path.join(PROCESSED_PATH, config['ingestion']['transactions_modeling_output']))
    user_logs = pd.read_parquet(os.path.join(PROCESSED_PATH, config['ingestion']['user_logs_output']))

    train_msno     = set(train['msno'])
    n_train        = len(train_msno)
    members_msno   = set(members['msno'])
    tx_msno        = set(tx['msno'])
    user_logs_msno = set(user_logs['msno'])

    del train, members, tx, user_logs
    gc.collect()

    groups = {
        'members':               train_msno & members_msno,
        'transactions_modeling': train_msno & tx_msno,
        'user_logs':             train_msno & user_logs_msno,
        'all_three_tables':      train_msno & members_msno & tx_msno & user_logs_msno,
        'no_table_match':        train_msno - members_msno - tx_msno - user_logs_msno,
    }

    return {
        name: {'users': len(s), 'pct': round(len(s) / n_train * 100, 1)}
        for name, s in groups.items()
    }


# ------------------------------------------------------------------------------
# Main — run full ingestion pipeline
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    split_user_logs_v1()
    aggregate_v1_splits()
    aggregate_v2()
    combine_user_logs()
    build_transactions()
    coverage = check_coverage()