# ==============================================================================
# Customer Intelligence Platform — Streamlit Dashboard
# ==============================================================================
# Entry point for the CIP portfolio dashboard. Currently contains the
# Executive Summary page. Additional pages (ROI Calculator, Segment Explorer,
# Churn Risk, Survival Curves) to be added in subsequent sessions.
#
# Run with: streamlit run streamlit_app.py
# ==============================================================================

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ------------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------------------

st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background-color: #0f1117;
        color: #e8eaf0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* ── KPI cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2540 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        transition: border-color 0.2s ease;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-card:hover { border-color: #4a90d9; }

    .kpi-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b7a9e;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.1rem;
        color: #e8eaf0;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.78rem;
        margin-top: 6px;
        color: #f0a070;
    }
    .kpi-delta.positive { color: #5cb85c; }

    /* ── Section headers ── */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #c8d0e8;
        margin-bottom: 4px;
    }
    .section-sub {
        font-size: 0.82rem;
        color: #6b7a9e;
        margin-bottom: 16px;
        line-height: 1.5;
    }

    /* ── Finding cards ── */
    .finding-card {
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
    }
    .finding-card.blue  { background: #132240; border-left: 3px solid #4a90d9; }
    .finding-card.amber { background: #241e10; border-left: 3px solid #d9a34a; }
    .finding-card.green { background: #102418; border-left: 3px solid #4a9d64; }

    .finding-title {
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .finding-title.blue  { color: #7db8f0; }
    .finding-title.amber { color: #f0c47d; }
    .finding-title.green { color: #7dc89a; }

    .finding-body {
        font-size: 0.82rem;
        color: #a0aabf;
        line-height: 1.55;
    }

    /* ── Divider ── */
    hr { border-color: #2a2f3e !important; }

    /* ── Chart containers ── */
    .chart-container {
        background: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 20px;
    }

    /* ── Page title ── */
    .page-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #e8eaf0;
        line-height: 1.15;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 0.88rem;
        color: #6b7a9e;
        letter-spacing: 0.05em;
    }

    /* ── Streamlit overrides ── */
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #a0aabf !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stMetricValue"] { color: #e8eaf0; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Config & paths
# ------------------------------------------------------------------------------

from pathlib import Path

# Use config.yaml if available (local dev), otherwise fall back to relative paths
_CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'

if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    PROCESSED_PATH = config['data']['processed_data_path']
    PLOTS_PATH     = config['data']['plots_path']
else:
    # Streamlit Cloud — data committed to repo at data/processed/
    _BASE_DIR      = Path(__file__).parent.parent
    PROCESSED_PATH = str(_BASE_DIR / 'data' / 'processed')
    PLOTS_PATH     = str(_BASE_DIR / 'data' / 'processed')

# Display name mapping — clean names for non-technical audience
SEGMENT_DISPLAY_NAMES = {
    0: 'Standard Loyal Users',
    1: 'Casual Users',
    2: 'Loyal Base',
    3: 'Annual Plan High Risk',
    4: 'Inactive Payers',
    5: 'Discount Hunting Users',
    6: 'At-Risk Renewers',
    7: 'Strong Long-Time Users',
    8: 'Ghost Users',
}

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

@st.cache_data
def load_data():
    ltv_preds = pd.read_parquet(os.path.join(PROCESSED_PATH, 'ltv_predictions.parquet'))
    seg_summary = pd.read_csv(os.path.join(PROCESSED_PATH, 'ltv_segment_summary.csv'))

    # Normalize column names — the CSV may have been saved from the ROI-augmented
    # dataframe (abbreviated names) rather than the clean segment summary.
    col_remap = {
        'churn_rate': 'actual_churn_rate',
        'avg_ltv':    'avg_ltv_12m',
        'total_ltv':  'total_ltv_12m',
    }
    seg_summary = seg_summary.rename(columns={k: v for k, v in col_remap.items()
                                               if k in seg_summary.columns})

    # Derive any columns the ROI save dropped
    if 'avg_ltv_12m_usd' not in seg_summary.columns:
        seg_summary['avg_ltv_12m_usd'] = (seg_summary['avg_ltv_12m'] / 30).round(2)
    if 'total_ltv_12m_usd' not in seg_summary.columns:
        seg_summary['total_ltv_12m_usd'] = (seg_summary['total_ltv_12m'] / 30).round(0)
    # median and avg_months not in ROI save — approximate from available columns
    if 'median_ltv_12m' not in seg_summary.columns:
        seg_summary['median_ltv_12m'] = seg_summary['avg_ltv_12m']
    if 'avg_months_alive' not in seg_summary.columns:
        seg_summary['avg_months_alive'] = (seg_summary['avg_ltv_12m'] / 149).round(2)

    # Apply display names
    ltv_preds['segment_display']   = ltv_preds['segment'].map(SEGMENT_DISPLAY_NAMES)
    seg_summary['segment_display'] = seg_summary['segment'].map(SEGMENT_DISPLAY_NAMES)

    # Ensure prob_churn_full exists (renamed from prob_calibrated in v1)
    if 'prob_calibrated' in ltv_preds.columns and 'prob_churn_full' not in ltv_preds.columns:
        ltv_preds = ltv_preds.rename(columns={'prob_calibrated': 'prob_churn_full'})

    return ltv_preds, seg_summary


ltv_preds, seg_summary = load_data()

# ------------------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------------------

st.sidebar.markdown("""
<div style='padding: 8px 0 20px 0;'>
    <div style='font-family: DM Serif Display, serif; font-size: 1.3rem; color: #c8d0e8;'>
        CIP Dashboard
    </div>
    <div style='font-size: 0.72rem; color: #6b7a9e; letter-spacing: 0.08em; text-transform: uppercase;'>
        KKBox Churn & LTV Analysis
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Executive Summary", "ROI & Survival Simulation", "Segment Explorer", "Model Deep Dive"],
    label_visibility="collapsed",
)

st.sidebar.markdown("<hr style='margin: 24px 0 20px 0; border-color: #2a2f3e;'>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='padding: 0 0 8px 0;'>
    <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
                text-transform: uppercase; color: #6b7a9e; margin-bottom: 14px;'>
        Built By
    </div>
    <div style='font-family: DM Serif Display, serif; font-size: 1.05rem;
                color: #c8d0e8; margin-bottom: 4px;'>
        Cameron Walker
    </div>
    <div style='font-size: 0.78rem; color: #6b7a9e; margin-bottom: 16px;'>
        Data Scientist
    </div>
    <a href='https://www.linkedin.com/in/cameronjwalker9/' target='_blank'
       style='display:flex; align-items:center; gap:8px; text-decoration:none;
              color:#a0aabf; font-size:0.82rem; margin-bottom:10px;
              transition: color 0.2s;'>
        <span style='font-size:1rem;'>🔗</span> LinkedIn
    </a>
    <a href='https://github.com/txcwalker' target='_blank'
       style='display:flex; align-items:center; gap:8px; text-decoration:none;
              color:#a0aabf; font-size:0.82rem; margin-bottom:10px;'>
        <span style='font-size:1rem;'>💻</span> GitHub
    </a>
    <a href='mailto:txcwalker@gmail.com'
       style='display:flex; align-items:center; gap:8px; text-decoration:none;
              color:#a0aabf; font-size:0.82rem; margin-bottom:10px;'>
        <span style='font-size:1rem;'>✉️</span> txcwalker@gmail.com
    </a>
</div>

<div style='margin-top: 24px; font-size: 0.70rem; color: #404660; line-height: 1.6;'>
    Built with Python · XGBoost · Streamlit<br>
    KKBox WSDM Cup 2018 dataset
</div>
""", unsafe_allow_html=True)
# ------------------------------------------------------------------------------
# Chart helpers
# ------------------------------------------------------------------------------

DARK_BG    = '#161b27'
GRID_COLOR = '#2a2f3e'
TEXT_COLOR = '#a0aabf'

def apply_dark_style(ax, fig):
    """Apply consistent dark theme to matplotlib figures."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


# Segment color palette — consistent across all charts
SEG_COLORS = {
    0: '#4a90d9',   # Standard Loyal Users     — blue
    1: '#7ec8a0',   # Casual Users              — mint
    2: '#5ab4d4',   # Loyal Base                — teal
    3: '#e05c5c',   # Annual Plan High Risk     — red
    4: '#d9a34a',   # Inactive Payers           — amber
    5: '#9b7ed9',   # Discount Hunting Users    — purple
    6: '#e07c3a',   # At-Risk Renewers          — orange
    7: '#4ad9b0',   # Strong Long-Time Users    — cyan-green
    8: '#888ea8',   # Ghost Users               — gray
}

# ------------------------------------------------------------------------------
# Executive Summary page
# ------------------------------------------------------------------------------

if page == "Executive Summary":

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='margin-bottom: 6px;'>
        <div class='page-title'>Customer Intelligence Platform</div>
        <div class='page-subtitle'>KKBOX MUSIC STREAMING &nbsp;·&nbsp; CHURN PREDICTION & LIFETIME VALUE ANALYSIS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0 24px 0;'>", unsafe_allow_html=True)

    # ── KPI calculations ──────────────────────────────────────────────────────────
    total_users_train = 970465  # full training population
    total_users_test = len(ltv_preds)
    churn_rate = ltv_preds['is_churn'].mean()
    avg_ltv_usd = ltv_preds['ltv_12m'].mean() / 30
    at_risk_mask = ltv_preds['prob_churn_full'] >= 0.30
    n_flagged = at_risk_mask.sum()

    # ── KPI row ───────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Overall Churn Rate</div>
            <div class='kpi-value'>{churn_rate:.1%}</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>Feb 2017 renewal snapshot</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Users Flagged at Risk</div>
            <div class='kpi-value'>{n_flagged:,}</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>≥30% churn probability</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Avg 12-Month LTV</div>
            <div class='kpi-value'>${avg_ltv_usd:.0f}</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>per user · test set</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Total Users Analyzed</div>
            <div class='kpi-value'>{total_users_train:,}</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>{total_users_test:,} held-out test set</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    # ── Chart 1: Churn rate by segment ────────────────────────────────────────
    with col1:
        st.markdown("<div class='section-header'>Churn Rate by Segment</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-sub'>How often customers in each segment cancel or fail to renew "
            "within 30 days of their membership expiring. The vertical dashed line is the overall snapshot churn rate"
            " of the dataset. Bars that go past this line represent the highest risk segments in terms of churning."
            ".</div>",
            unsafe_allow_html=True,
        )

        plot_df = seg_summary.sort_values('actual_churn_rate', ascending=True)
        colors  = [SEG_COLORS.get(int(s), '#888ea8') for s in plot_df['segment']]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        apply_dark_style(ax, fig)

        bars = ax.barh(
            plot_df['segment_display'],
            plot_df['actual_churn_rate'],
            color=colors,
            height=0.62,
            edgecolor='none',
        )

        overall = ltv_preds['is_churn'].mean()
        ax.axvline(overall, color='#e8eaf0', linewidth=1.0, linestyle='--', alpha=0.5,
                   label=f'Overall avg ({overall:.1%})')

        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlabel('Churn Rate', fontsize=8)

        # Value labels
        for bar, val in zip(bars, plot_df['actual_churn_rate']):
            ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1%}', va='center', fontsize=7.5, color=TEXT_COLOR)

        ax.legend(fontsize=7.5, framealpha=0, labelcolor=TEXT_COLOR)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig)
        plt.close()

    # ── Chart 2: Average LTV by segment ───────────────────────────────────────
    with col2:
        st.markdown("<div class='section-header'>Average 12-Month LTV by Segment</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-sub'>Projected average revenue per user per segment over the next 12 months "
            "based on Monte Carlo survival simulation. Finding the high churn risk users in the high LTV segments is "
            "one of the best opportunities for recovering potentially lost revenue. For more information on the "
            "segments see the segment explorer page."
            "</div>",
            unsafe_allow_html=True,
        )

        plot_df2 = seg_summary.sort_values('avg_ltv_12m', ascending=True)
        colors2  = [SEG_COLORS.get(int(s), '#888ea8') for s in plot_df2['segment']]
        ltv_usd  = plot_df2['avg_ltv_12m'] / 30

        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        apply_dark_style(ax2, fig2)

        bars2 = ax2.barh(
            plot_df2['segment_display'],
            ltv_usd,
            color=colors2,
            height=0.62,
            edgecolor='none',
        )

        ax2.set_xlabel('Avg 12-Month LTV (USD)', fontsize=8)

        for bar, val in zip(bars2, ltv_usd):
            ax2.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     f'${val:.0f}', va='center', fontsize=7.5, color=TEXT_COLOR)

        fig2.tight_layout(pad=1.2)
        st.pyplot(fig2)
        plt.close()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0 0 24px 0;'>", unsafe_allow_html=True)

    # ── Key findings row ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header' style='margin-bottom:16px;'>Key Findings</div>",
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="medium")

    with f1:
        st.markdown("""
        <div class='finding-card blue'>
            <div class='finding-title blue'>🔵 &nbsp;Behavior Predicts Churn Better Than Demographics</div>
            <div class='finding-body'>
                A behavioral-only model achieved <strong style='color:#c8d0e8;'>0.897 AUROC</strong>.
                Adding demographic features such as age, gender and city improved the model by just
                <strong style='color:#c8d0e8;'>+0.004.</strong> This suggests that
                <em>customer's actions</em> matters more than their person details.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class='finding-card amber'>
            <div class='finding-title amber'>🟡 &nbsp;One Segment Drives the Intervention Case</div>
            <div class='finding-body'>
                <strong style='color:#c8d0e8;'>At-Risk Renewers</strong> churn at 35% but carry
                meaningful lifetime value. They are actively choosing not to auto-renew each cycle
                rather than abandoning the service. Our model found that the most important feature in predicting churn is
                whether or not the customer used the auto renew feature to keep their membership. This makes them the most
                recoverable segment and gives a clear plan to convert these users to a more long term segment. 
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class='finding-card green'>
            <div class='finding-title green'>🟢 &nbsp;Most Revenue Is Stable</div>
            <div class='finding-body'>
                The three largest segments are the <strong style='color:#c8d0e8;'>Standard Loyal Users,
                Casual Users, and Strong Long-Time Users</strong>. They represent over
                <strong style='color:#c8d0e8;'>60% of the user base</strong> and all of whom have churn rates
                below 8%. Their projected 12-month survival exceeds 10 months on average,
                anchoring the overall LTV picture. 
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

    # ── Model footnote ────────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-size: 0.72rem; color: #404660; text-align: center; letter-spacing: 0.04em;'>
        XGBoost churn model trained on 970,465 users &nbsp;·&nbsp;
        Weibull Monte Carlo LTV (k=0.95, 1,000 simulations, 12-month time horizon) &nbsp;·&nbsp;
        KKBox WSDM Cup 2018 dataset &nbsp;·&nbsp; All amounts in USD (1 USD ≈ 30 NTD)
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# Page 2 — ROI & Survival Simulation
# ==============================================================================

elif page == "ROI & Survival Simulation":

    st.markdown("""
    <div style='margin-bottom: 6px;'>
        <div class='page-title'>ROI & Survival Simulation</div>
        <div class='page-subtitle'>INTERVENTION MODELING &nbsp;</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0 20px 0;'>", unsafe_allow_html=True)

    # ── How it works explainer ────────────────────────────────────────────────
    st.markdown("""
    <div style='background: #1a2035; border: 1px solid #2a3050; border-radius: 12px;
                padding: 20px 24px; margin-bottom: 24px;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.1rem;
                    color: #c8d0e8; margin-bottom: 10px;'>How the simulation works</div>
        <div style='font-size: 0.83rem; color: #a0aabf; line-height: 1.7;'>
            Each user has a churn probability from the model and is used to simulate <strong style='color:#c8d0e8;'>
            1,000 futures</strong> over the next 12 months. In each simulation, the user either stays or leaves each
            month based on their individual churn probability assigned by the model. There is 2.5% churn each month to
            simulate the randomness of the real world, These users represent attrition outside of the reach of any
             campaign(things like job loss, moving, loss of interest, etc).
            If we average a users outcomes over the 1000 simulations we can calculate their
            <strong style='color:#c8d0e8;'>Lifetime Value (LTV) </strong>. The ROI calculator below uses these LTV
            estimates to answer a practical question: <em style='color:#7db8f0;'>If we reach out to at-risk users,
            how much revenue do we earn relative to what we spend?</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0 16px 0; border-color: #2a2f3e;'>", unsafe_allow_html=True)

    # ── How many users are in the pool ───────────────────────────────────────────
    total_flagged = int(seg_summary['n_flagged'].sum())
    st.markdown(f"""
    <div style='background:#1a2035; border:1px solid #2a3050; border-radius:10px;
                padding:14px 20px; margin-bottom:20px; font-size:0.83rem; color:#a0aabf;'>
        <strong style='color:#c8d0e8;'>The model flagged {total_flagged:,} users</strong> as at risk because their
        model-predicted churn probability exceeds 30%.  This probability is adjustable (not here) depending on how many unlikely
        low risk churn users you are willing to contact to reach more high risk users. The lower the probability the
        more of both group you would potentially contact. The reach rate below determines what fraction of this pool
        actually receives an outreach message.
    </div>
    """, unsafe_allow_html=True)

    # ── ROI sliders ───────────────────────────────────────────────────────────────
    sl1, sl2, sl3 = st.columns(3)
    with sl1:
        reach_pct = st.slider(
            "Reach Rate — % of flagged users contacted",
            min_value=0, max_value=100, value=15, step=5,
            format="%d%%",
            help="What fraction of the users we flag actually receive an outreach message.",
        )
        reach_rate = reach_pct / 100

    with sl2:
        response_pct = st.slider(
            "Response Rate — % of contacted users retained",
            min_value=0, max_value=100, value=30, step=5,
            format="%d%%",
            help="Of the users we contact, what fraction successfully renew as a result.",
        )
        response_rate = response_pct / 100

    with sl3:
        cost_per_user = st.slider(
            "Cost per User Contacted (USD)",
            min_value=1, max_value=20, value=3, step=1,
            help="Fully-loaded cost of one outreach touchpoint — email, push notification, or discount.",
        )

    # ── ROI calculation ───────────────────────────────────────────────────────
    roi_df = seg_summary.copy()
    cost_per_user_ntd = cost_per_user * 30

    roi_df['n_contacted']       = (roi_df['n_flagged'] * reach_rate).astype(int)
    roi_df['n_saved']           = (roi_df['n_contacted'] * response_rate).astype(int)
    roi_df['revenue_saved_usd'] = (roi_df['n_saved'] * roi_df['avg_ltv_12m'] / 30).round(0)
    roi_df['cost_usd']          = (roi_df['n_contacted'] * cost_per_user_ntd / 30).round(0)
    roi_df['net_roi_usd']       = roi_df['revenue_saved_usd'] - roi_df['cost_usd']
    roi_df['roi_multiple']      = (
        roi_df['revenue_saved_usd'] / roi_df['cost_usd'].replace(0, np.nan)
    ).round(2).fillna(0)

    roi_df = roi_df.sort_values('net_roi_usd', ascending=False)

    # ── ROI summary KPIs ──────────────────────────────────────────────────────
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    total_contacted = roi_df['n_contacted'].sum()
    total_saved     = roi_df['n_saved'].sum()
    total_rev_saved = roi_df['revenue_saved_usd'].sum()
    total_cost      = roi_df['cost_usd'].sum()
    total_net       = roi_df['net_roi_usd'].sum()
    overall_multiple = total_rev_saved / total_cost if total_cost > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Users Contacted</div>
            <div class='kpi-value'>{total_contacted:,}</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>{reach_pct}% of {total_flagged:,} flagged users</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Users Retained</div>
            <div class='kpi-value'>{total_saved:,}</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>{response_pct}% response rate assumed</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Net Revenue Saved</div>
            <div class='kpi-value'>${total_net:,.0f}</div>
            <div class='kpi-delta {"positive" if total_net > 0 else ""}'>
                ${total_rev_saved:,.0f} saved · ${total_cost:,.0f} spent
            </div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Overall ROI Multiple</div>
            <div class='kpi-value'>{overall_multiple:.1f}×</div>
            <div class='kpi-delta {"positive" if overall_multiple >= 1 else ""}'>
                {"above break-even" if overall_multiple >= 1 else "below break-even"}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # ── ROI charts ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='section-header'>Net ROI by Segment</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Total revenue saved minus outreach cost. "
                    "Green = positive return, red = spending more than you save.</div>",
                    unsafe_allow_html=True)

        plot_roi = roi_df.sort_values('net_roi_usd', ascending=True)
        bar_colors = ['#4a9d64' if v >= 0 else '#c0392b' for v in plot_roi['net_roi_usd']]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        apply_dark_style(ax, fig)
        ax.barh(plot_roi['segment_display'], plot_roi['net_roi_usd'],
                color=bar_colors, height=0.62, edgecolor='none')
        ax.axvline(0, color='#e8eaf0', linewidth=0.8, alpha=0.4)
        ax.set_xlabel('Net ROI (USD)', fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        for bar, val in zip(ax.patches, plot_roi['net_roi_usd']):
            offset = 50 if val >= 0 else -50
            ha = 'left' if val >= 0 else 'right'
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    f'${val:,.0f}', va='center', ha=ha, fontsize=7.5, color=TEXT_COLOR)

        fig.tight_layout(pad=1.2)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-header'>ROI Multiple by Segment</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Revenue saved per dollar spent. "
                    "Above 1× means the intervention pays for itself.</div>",
                    unsafe_allow_html=True)

        plot_mul = roi_df.sort_values('roi_multiple', ascending=True)
        mul_colors = ['#4a9d64' if v >= 1 else '#c0392b' for v in plot_mul['roi_multiple']]

        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        apply_dark_style(ax2, fig2)
        ax2.barh(plot_mul['segment_display'], plot_mul['roi_multiple'],
                 color=mul_colors, height=0.62, edgecolor='none')
        ax2.axvline(1, color='#e8eaf0', linewidth=1.0, linestyle='--', alpha=0.5,
                    label='Break-even (1×)')
        ax2.set_xlabel('ROI Multiple', fontsize=8)

        for bar, val in zip(ax2.patches, plot_mul['roi_multiple']):
            ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1f}×', va='center', fontsize=7.5, color=TEXT_COLOR)

        ax2.legend(fontsize=7.5, framealpha=0, labelcolor=TEXT_COLOR)
        fig2.tight_layout(pad=1.2)
        st.pyplot(fig2)
        plt.close()

    # ── ROI detail table ──────────────────────────────────────────────────────
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    with st.expander("Full ROI breakdown by segment"):
        table_cols = {
            'segment_display':  'Segment',
            'n_flagged':        'Flagged',
            'n_contacted':      'Contacted',
            'n_saved':          'Retained',
            'avg_ltv_12m_usd':  'Avg LTV (USD)',
            'revenue_saved_usd':'Revenue Saved',
            'cost_usd':         'Cost',
            'net_roi_usd':      'Net ROI',
            'roi_multiple':     'Multiple',
        }
        tdf = roi_df[list(table_cols.keys())].rename(columns=table_cols).copy()
        tdf['Revenue Saved'] = tdf['Revenue Saved'].map('${:,.0f}'.format)
        tdf['Cost']          = tdf['Cost'].map('${:,.0f}'.format)
        tdf['Net ROI']       = tdf['Net ROI'].map('${:,.0f}'.format)
        tdf['Multiple']      = tdf['Multiple'].map('{:.1f}×'.format)
        st.dataframe(tdf, use_container_width=True, hide_index=True)


# ==============================================================================
# Page 3 — Segment Explorer
# ==============================================================================

elif page == "Segment Explorer":

    st.markdown("""
    <div style='margin-bottom: 6px;'>
        <div class='page-title'>Segment Explorer</div>
        <div class='page-subtitle'>CUSTOMER SEGMENTATION &nbsp;·&nbsp; WHO THEY ARE AND WHY IT MATTERS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0 20px 0;'>", unsafe_allow_html=True)

    # ── Explainer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background: #1a2035; border: 1px solid #2a3050; border-radius: 12px;
                padding: 20px 24px; margin-bottom: 28px;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.1rem;
                    color: #c8d0e8; margin-bottom: 10px;'>How customers were grouped</div>
        <div style='font-size: 0.83rem; color: #a0aabf; line-height: 1.7;'>
            Rather than defining customer types by hand, we let the data reveal its own natural groupings.
            The algorithm looked at <strong style='color:#c8d0e8;'>27 behavioral signals</strong> for each
            of the 970,000 users — things like how long they've been subscribed, whether they use auto-renewal,
            how recently they listened, and whether they've ever cancelled — and found
            <strong style='color:#c8d0e8;'>8 clusters</strong> of users who behave similarly to each other
            and differently from everyone else. Demographics like age and city were intentionally excluded:
            they added no meaningful separation and would make the model less generalizable.
            Each segment below represents a genuinely distinct customer archetype with its own
            retention risk and revenue profile.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Segment definitions ───────────────────────────────────────────────────
    # Written from feature profile knowledge — tidy as needed
    SEGMENT_PROFILES = {
        0: {
            'icon': '🔵',
            'color': '#4a90d9',
            'tagline': 'Reliable payers with no listening history on record.',
            'description': (
                'These users have consistent transaction records and stable renewal behavior '
                'but no engagement data — they pay reliably and quietly. Low churn suggests '
                'they are satisfied subscribers who simply don\'t interact with the app heavily. '
                'Their absence from the listening logs is a data pattern, not a sign of disengagement.'
            ),
            'signals': ['Consistent payment history', 'Auto-renewal on', 'No log activity recorded'],
        },
        1: {
            'icon': '🟢',
            'color': '#7ec8a0',
            'tagline': 'Light users who stay subscribed at low cost.',
            'description': (
                'The largest segment. Casual users who listen infrequently, '
                'but they churn at the lowest rate of any group. They represent a stable, '
                'low-maintenance base. Probably unlikely to respond to engagement campaigns but also '
                'unlikely to leave without a reason.'
            ),
            'signals': ['Low listening activity', 'Short plan durations', 'Very low churn'],
        },
        2: {
            'icon': '🩵',
            'color': '#5ab4d4',
            'tagline': 'Steady mid-tier subscribers with strong retention.',
            'description': (
                'A dependable core segment with moderate engagement and consistent renewals. '
                'They have been subscribers for a meaningful period and show no signs of '
                'declining activity. Solid LTV with minimal intervention required.'
            ),
            'signals': ['Moderate engagement', 'Regular renewal pattern', 'Mid-range tenure'],
        },
        3: {
            'icon': '🔴',
            'color': '#e05c5c',
            'tagline': 'Annual plan holders approaching the end of their cycle.',
            'description': (
                'The highest-churn segment by a wide margin. These users are on long-duration '
                'plans that are expiring — when an annual plan ends, many simply don\'t renew. '
                'Their very low LTV reflects that most are already on the way out. '
                'Intervention cost exceeds recoverable revenue at typical response rates.'
            ),
            'signals': ['Annual plan duration', 'Manual renewal only', '86% churn rate'],
        },
        4: {
            'icon': '🟡',
            'color': '#d9a34a',
            'tagline': 'Paying subscribers who have gone quiet.',
            'description': (
                'These users have valid payment records and have never cancelled, but their '
                'listening activity has dropped off in the last six months. They are paying '
                'out of habit rather than active use. This pattern often precedes eventual '
                'cancellation. Worth monitoring but not yet high-risk.'
            ),
            'signals': ['No recent listening activity', 'Never cancelled', 'Active payment history'],
        },
        5: {
            'icon': '🟣',
            'color': '#9b7ed9',
            'tagline': 'Value-seekers who respond to promotions.',
            'description': (
                'A smaller but distinct group with an above-average share of discounted '
                'transactions. They are price-aware subscribers who have historically taken '
                'advantage of promotional pricing. Good LTV suggests they stick around when '
                'the value proposition is right.'
            ),
            'signals': ['High discount transaction rate', 'Price-sensitive renewal pattern', 'Solid tenure'],
        },
        6: {
            'icon': '🟠',
            'color': '#e07c3a',
            'tagline': 'The primary retention opportunity.',
            'description': (
                'At-Risk Renewers are the most actionable segment in the dataset. They churn '
                'at 35%, well above average, but carry meaningful lifetime value because '
                'they are not abandoning the service. They are simply choosing not to auto-renew '
                'each cycle. A targeted nudge at the right moment has the highest probability '
                'of changing their decision.'
            ),
            'signals': ['Manual renewal only', 'Moderate tenure', 'Recent activity dropping off'],
        },
        7: {
            'icon': '🩵',
            'color': '#4ad9b0',
            'tagline': 'Long-tenure, high-value loyalists.',
            'description': (
                'The highest-value segment per user. These subscribers have been around the '
                'longest, listen consistently, and renew reliably. They are the foundation '
                'of the business — low churn, high engagement, and strong projected LTV. '
            ),
            'signals': ['Longest average tenure', 'High engagement metrics', 'Consistent auto-renewal'],
        },
        8: {
            'icon': '⚫',
            'color': '#888ea8',
            'tagline': 'Users with no transaction, demographic, or listening data.',
            'description': (
                'Ghost users registered but left no trace across any data source. '
                'With no payment history, no listening activity, and no demographic record, '
                'there is no behavioral signal to model. They were excluded from clustering '
                'and assigned their own group. Their high churn rate reflects users who '
                'signed up and never meaningfully engaged with the service.'
            ),
            'signals': ['No transaction record', 'No listening history', 'No demographic data'],
        },
    }

    # ── Merge profile data with segment summary ────────────────────────────────
    # Build display rows: 3 columns × 3 rows
    seg_rows = seg_summary.sort_values('segment').to_dict('records')

    def render_segment_card(row, profile):
        churn_pct  = row['actual_churn_rate'] * 100
        ltv_usd    = row['avg_ltv_12m'] / 30
        n_users    = int(row['n_users'])
        pct_base   = n_users / seg_summary['n_users'].sum() * 100
        color      = profile['color']
        signals_html = ''.join(
            f"<span style='display:inline-block; background:#1e2540; border:1px solid #2a3050; "
            f"border-radius:20px; padding:3px 10px; margin:3px 3px 0 0; font-size:0.72rem; "
            f"color:#a0aabf;'>{s}</span>"
            for s in profile['signals']
        )
        return f"""
        <div style='background: linear-gradient(135deg, #1a2035 0%, #1e2540 100%);
                    border: 1px solid #2a3050; border-left: 3px solid {color};
                    border-radius: 12px; padding: 18px 20px; height: 100%;
                    margin-bottom: 4px;'>
            <div style='display:flex; align-items:baseline; gap:10px; margin-bottom:6px;'>
                <span style='font-size:1.3rem;'>{profile['icon']}</span>
                <span style='font-family: DM Serif Display, serif; font-size:1.05rem;
                             color:#c8d0e8;'>{row['segment_display']}</span>
            </div>
            <div style='font-size:0.78rem; color:{color}; font-weight:600;
                        margin-bottom:10px; font-style:italic;'>{profile['tagline']}</div>
            <div style='font-size:0.78rem; color:#8090b0; line-height:1.6;
                        margin-bottom:12px;'>{profile['description']}</div>
            <div style='margin-bottom:12px;'>{signals_html}</div>
            <div style='display:flex; gap:20px; border-top:1px solid #2a3050;
                        padding-top:12px; margin-top:4px;'>
                <div>
                    <div style='font-size:0.68rem; color:#6b7a9e; text-transform:uppercase;
                                letter-spacing:0.1em;'>Churn Rate</div>
                    <div style='font-family: DM Serif Display, serif; font-size:1.2rem;
                                color:{"#e05c5c" if churn_pct > 20 else "#c8d0e8"};'>
                        {churn_pct:.1f}%</div>
                </div>
                <div>
                    <div style='font-size:0.68rem; color:#6b7a9e; text-transform:uppercase;
                                letter-spacing:0.1em;'>Avg LTV</div>
                    <div style='font-family: DM Serif Display, serif; font-size:1.2rem;
                                color:#c8d0e8;'>${ltv_usd:.0f}</div>
                </div>
                <div>
                    <div style='font-size:0.68rem; color:#6b7a9e; text-transform:uppercase;
                                letter-spacing:0.1em;'>Users</div>
                    <div style='font-family: DM Serif Display, serif; font-size:1.2rem;
                                color:#c8d0e8;'>{n_users:,}
                        <span style='font-size:0.72rem; color:#6b7a9e;'>({pct_base:.1f}%)</span>
                    </div>
                </div>
            </div>
        </div>
        """

    # Render grid 3 columns wide
    for row_start in range(0, len(seg_rows), 3):
        cols = st.columns(3, gap="medium")
        for col_idx, col in enumerate(cols):
            data_idx = row_start + col_idx
            if data_idx < len(seg_rows):
                row     = seg_rows[data_idx]
                seg_id  = int(row['segment'])
                profile = SEGMENT_PROFILES.get(seg_id, {
                    'icon': '⚪', 'color': '#888ea8',
                    'tagline': '', 'description': '', 'signals': [],
                })
                with col:
                    st.markdown(render_segment_card(row, profile),
                                unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


    # ── PCA visualization ─────────────────────────────────────────────────────
    st.markdown("<hr style='margin: 28px 0 24px 0;'>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Cluster Separation — PCA Projection</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>The graph below attempts to visualize how the clusters are laid out across space."
        "The math here is complicated but it effectively compresses the 27 features used in the model down to 2 and"
        " visualizes them in 2-dimensional space. Overlap in this view is expected and does not mean the clusters "
        "are poorly defined, the two axes account for about 44% of the variance in the model. In the full 27-dimensional"
        " space the segments are more cleanly separated than this projection suggests. It is just much harder to"
        " visualize anything in 27 dimensional space</div>",
        unsafe_allow_html=True,
    )

    pca_path = os.path.join(config['data']['plots_path'], 'pca_cluster_viz.png')
    if os.path.exists(pca_path):
        pca_img = plt.imread(pca_path)
        fig_pca, ax_pca = plt.subplots(figsize=(12, 6))
        fig_pca.patch.set_facecolor(DARK_BG)
        ax_pca.set_facecolor(DARK_BG)
        ax_pca.imshow(pca_img)
        ax_pca.axis('off')
        fig_pca.tight_layout(pad=0)
        st.pyplot(fig_pca)
        plt.close()
    else:
        # PCA plot not found — show segment map as fallback
        st.info(
            "PCA plot not found. Re-run 03_segmentation.py to generate pca_cluster_viz.png. "
            "Showing segment map (churn risk vs LTV) as fallback."
        )
        fig_pca, ax_pca = plt.subplots(figsize=(10, 5))
        apply_dark_style(ax_pca, fig_pca)
        for _, row in seg_summary.iterrows():
            seg_id = int(row['segment'])
            color  = SEG_COLORS.get(seg_id, '#888ea8')
            ax_pca.scatter(
                row['avg_churn_prob'], row['avg_ltv_12m'] / 30,
                s=row['n_users'] / 30, color=color, alpha=0.85,
                edgecolors='white', linewidths=0.5,
            )
            ax_pca.annotate(
                row['segment_display'],
                (row['avg_churn_prob'], row['avg_ltv_12m'] / 30),
                textcoords='offset points', xytext=(8, 4),
                fontsize=7.5, color=TEXT_COLOR,
            )
        ax_pca.set_xlabel('Avg Churn Probability', fontsize=9, color=TEXT_COLOR)
        ax_pca.set_ylabel('Avg LTV (USD)', fontsize=9, color=TEXT_COLOR)
        ax_pca.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        fig_pca.tight_layout(pad=1.2)
        st.pyplot(fig_pca)
        plt.close()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ── Technical callout ─────────────────────────────────────────────────────
    st.markdown("<hr style='margin: 28px 0 20px 0;'>", unsafe_allow_html=True)

    with st.expander("Technical details — how the clustering was built"):
        st.markdown("""
        <div style='font-size: 0.82rem; color: #a0aabf; line-height: 1.8;'>
            <strong style='color:#c8d0e8;'>Algorithm:</strong> K-Means clustering on 968,436 users
            with transaction data. Ghost users (no transactions) excluded from clustering and assigned
            their own segment label.<br><br>
            <strong style='color:#c8d0e8;'>Feature set:</strong> 27 features — 22 behavioral
            (subscription structure, payment behavior, renewal patterns, engagement metrics) + 5
            structural missingness flags. Demographics excluded: age, gender, city and registration
            channel produced monotonically worse separation scores at every k value tested.<br><br>
            <strong style='color:#c8d0e8;'>Scaling:</strong> Winsorize at 1st/99th percentile →
            StandardScaler. RobustScaler was ruled out — 10 of 22 features have zero IQR due to
            heavily modal distributions (e.g. 95% of users on 30-day plans), causing divide-by-zero
            and scaled values that dominate distance calculations.<br><br>
            <strong style='color:#c8d0e8;'>K-selection:</strong> 4 feature set variants × k=3–15
            evaluated on inertia, silhouette score, Davies-Bouldin index, and Calinski-Harabasz score.
            DB minimum at k=7, Calinski-Harabasz peak at k=8, silhouette plateaus after k=10.
            <strong style='color:#c8d0e8;'>k=8 selected</strong> — two of three primary metrics
            point directly to it, and 8 segments is the largest number that remains interpretable
            for a business audience.<br><br>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# Page 4 — Model Deep Dive
# ==============================================================================

elif page == "Model Deep Dive":

    st.markdown("""
    <div style='margin-bottom: 6px;'>
        <div class='page-title'>Model Deep Dive</div>
        <div class='page-subtitle'>XGBOOST CHURN MODEL &nbsp;·&nbsp; FEATURE ENGINEERING &amp; EVALUATION</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0 20px 0;'>", unsafe_allow_html=True)

    # ── Model performance summary ─────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-label'>Behavioral AUROC</div>
            <div class='kpi-value'>0.897</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>27 features · no demographics</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-label'>Training Population</div>
            <div class='kpi-value'>970,465</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>users · 80/10/10 split</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-label'>Demographic AUROC Lift</div>
            <div class='kpi-value'>+0.004</div>
            <div class='kpi-delta' style='color:#6b7a9e;'>4 extra features · negligible gain</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-label'>Val → Test Gap</div>
            <div class='kpi-value'>−0.001</div>
            <div class='kpi-delta positive'>no overfitting</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # ── Feature engineering section ───────────────────────────────────────────
    st.markdown("<div class='section-header'>Key Engineered Features</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'> Some of the features in the model were used directly from the dataset, others "
        "were engineered from the data in order to showcase meaningful relationships between variables more directly."
        "Below is a review of some of the more key engineered features in predicting churn."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    fe1, fe2 = st.columns(2, gap="large")

    with fe1:
        st.markdown("""
        <div style='background:#1a2035; border:1px solid #2a3050; border-left:3px solid #4a90d9;
                    border-radius:12px; padding:18px 20px; margin-bottom:14px;'>
            <div style='font-weight:600; color:#7db8f0; font-size:0.88rem;
                        margin-bottom:8px;'>Auto Renew Delta</div>
            <div style='font-size:0.80rem; color:#a0aabf; line-height:1.65;'>
                The difference between a user's <em>current</em> auto-renewal setting and their
                <em>historical average</em>. A negative delta indicates they recently switched off
                auto-renew after a long history of keeping it on. In all versions of the model this proved to be one
                of the strongest single signals of impending churn.
            </div>
        </div>

        <div style='background:#1a2035; border:1px solid #2a3050; border-left:3px solid #d9a34a;
                    border-radius:12px; padding:18px 20px; margin-bottom:14px;'>
            <div style='font-weight:600; color:#f0c47d; font-size:0.88rem;
                        margin-bottom:8px;'>Number of Transactions in the last Six Months</div>
            <div style='font-size:0.80rem; color:#a0aabf; line-height:1.65;'>
                A 6-month activity window built from the available transactions. Users absent from
                this window get <code style='background:#0f1117; padding:1px 5px;
                border-radius:3px; font-size:0.76rem;'>n_transactions_6m = −1</code>
                rather than 0 — distinguishing "not present in window" from
                "present but no activity." The flag makes this distinction explicit for the model.
            </div>
        </div>

        <div style='background:#1a2035; border:1px solid #2a3050; border-left:3px solid #4a9d64;
                    border-radius:12px; padding:18px 20px; margin-bottom:14px;'>
            <div style='font-weight:600; color:#7dc89a; font-size:0.88rem;
                        margin-bottom:8px;'>Total Membership days</div>
            <div style='font-size:0.80rem; color:#a0aabf; line-height:1.65;'>
                True paid tenure is calculated by merging overlapping subscription intervals before summing. The
                transactions table contains same-day duplicate records and overlapping plan periods — summing raw
                plan durations without collapsing these would double-count days and overstate how long a user has
                actually been subscribed.
            </div>
        </div>""", unsafe_allow_html=True)

    with fe2:
        st.markdown("""
        <div style='background:#1a2035; border:1px solid #2a3050; border-left:3px solid #9b7ed9;
                    border-radius:12px; padding:18px 20px; margin-bottom:14px;'>
            <div style='font-weight:600; color:#c4a8f0; font-size:0.88rem;
                        margin-bottom:8px;'>completion_rate &amp; skip_rate</div>
            <div style='font-size:0.80rem; color:#a0aabf; line-height:1.65;'>
                Derived from raw play counts in the listening logs.
                <code style='background:#0f1117; padding:1px 5px; border-radius:3px;
                font-size:0.76rem;'>completion_rate</code> = full plays ÷ total plays;
                <code style='background:#0f1117; padding:1px 5px; border-radius:3px;
                font-size:0.76rem;'>skip_rate</code> = abandoned-in-first-25% ÷ total plays.
                Together they capture engagement quality, not just quantity. A user
                who plays many songs but skips most of them is behaviorally different
                from one who listens all the way through.
            </div>
        </div>

        <div style='background:#1a2035; border:1px solid #2a3050; border-left:3px solid #e07c3a;
                    border-radius:12px; padding:18px 20px; margin-bottom:14px;'>
            <div style='font-weight:600; color:#f0a870; font-size:0.88rem;
                        margin-bottom:8px;'>Missingness flags</div>
            <div style='font-size:0.80rem; color:#a0aabf; line-height:1.65;'>
                Three binary indicators —
                <code style='background:#0f1117; padding:1px 5px; border-radius:3px;
                font-size:0.76rem;'>has_no_transactions</code>,
                <code style='background:#0f1117; padding:1px 5px; border-radius:3px;
                font-size:0.76rem;'>has_no_logs</code>,
                <code style='background:#0f1117; padding:1px 5px; border-radius:3px;
                font-size:0.76rem;'>has_no_demographics</code> all built before any imputation runs.
                Absence from a source table is itself a predictive signal.
                Imputing silently and discarding the flag would let the model see
                filled-in values without knowing they were filled in. It should be noted that only 495 users did not 
                have transactions available and they were cut from the prediction data (along with this flag afterward).
                The remaining flags were put in place to flag users who either did not want to or had not yet updated 
                their personal information (although most of this was not used in the model since it was not
                 meaningfully helpful). 
            </div>
        </div>

        <div style='background:#1a2035; border:1px solid #2a3050; border-left:3px solid #5ab4d4;
                    border-radius:12px; padding:18px 20px; margin-bottom:14px;'>
            <div style='font-weight:600; color:#8ad4ee; font-size:0.88rem;
                        margin-bottom:8px;'>Feature cutoff — Feb 28, 2017</div>
            <div style='font-size:0.80rem; color:#a0aabf; line-height:1.65;'>
                All behavioral features are built from data available through February 28, 2017.
                Churn labels are determined by whether a user renewed following their February
                expiration date. Using data from the same month as the labels ensures the model
                reflects what was actually knowable at prediction time — no future information
                is used to predict a present outcome.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 28px 0 24px 0;'>", unsafe_allow_html=True)

    # ── Calibration section ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>How Calibration and AUROC Work Together</div>",
                unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    cal_text, cal_plot = st.columns([1, 1], gap="large")

    with cal_text:
        st.markdown("""
        <div style='background:#1a2035; border:1px solid #2a3050; border-radius:12px;
                    padding:20px 22px; height:100%;'>
            <div style='font-size:0.82rem; color:#a0aabf; line-height:1.8;'>
                With a 9% churn rate, the standard approach is to upweight churners during
                training so the model pays more attention to the minority class and is penalized more for misses
                preventing the model chasing 91% accuracy and predicting everything as Non Churn. This was the case 
                for the first version of the model, it pushed AUROC close to perfect on the validation set.<br><br>
                The problem is the predicted probabilities became severely overconfident.
                Every segment showed predicted churn rates 15–30 percentage points above
                actual rates. A model that says "40% chance of churn" when the true rate
                is 10% is not useful for any reason including for additional tools such as a LTV simulation
                or ROI calculator, both of which use the raw probability output directly.<br><br>
                Removing the class weighting changed
                <strong style='color:#c8d0e8;'>essentially nothing in AUROC</strong>
                and produced
                <strong style='color:#c8d0e8;'>a near-perfect calibration</strong> —
                segment predicted vs actual churn deltas all under 0.5%.
                A model needs to be both accurate and well calibrated to be useful and trustworthy.
                It is just one tool in the toolbox of the decision making process. 
            </div>
        </div>""", unsafe_allow_html=True)

    with cal_plot:
        from sklearn.calibration import calibration_curve
        fig_cal, ax_cal = plt.subplots(figsize=(6, 4.5))
        apply_dark_style(ax_cal, fig_cal)

        for label, col in [('Behavioral', 'prob_churn_behavioral'), ('Full', 'prob_churn_full')]:
            if col in ltv_preds.columns:
                frac_pos, mean_pred = calibration_curve(
                    ltv_preds['is_churn'], ltv_preds[col], n_bins=10
                )
                ax_cal.plot(mean_pred, frac_pos, marker='o', linewidth=2,
                            label=label, color=SEG_COLORS.get(0 if label == 'Behavioral' else 2))

        ax_cal.plot([0, 1], [0, 1], linestyle='--', linewidth=1,
                    color='#e8eaf0', alpha=0.4, label='Perfect calibration')
        ax_cal.set_xlabel('Mean predicted probability', fontsize=8)
        ax_cal.set_ylabel('Actual churn rate', fontsize=8)
        ax_cal.set_title('Calibration Curves', fontsize=10, color=TEXT_COLOR, pad=8)
        ax_cal.legend(fontsize=8, framealpha=0, labelcolor=TEXT_COLOR)
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        fig_cal.tight_layout(pad=1.2)
        st.pyplot(fig_cal)
        plt.close()

    st.markdown("<hr style='margin: 28px 0 24px 0;'>", unsafe_allow_html=True)

    # ── SHAP / feature importance ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>What Drives the Predictions</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>The chart below ranks each feature by its average contribution "
        "to the model's predictions across the test set. Features at the top had the most influence "
        "on whether a user was flagged as likely to churn.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    shap_bee = os.path.join(PROCESSED_PATH, 'shap_beeswarm.png')
    shap_bar = os.path.join(PROCESSED_PATH, 'shap_bar.png')

    if os.path.exists(shap_bar):
        sh_img = plt.imread(shap_bar)
        fig_sh, ax_sh = plt.subplots(figsize=(7, 5))
        fig_sh.patch.set_facecolor(DARK_BG)
        ax_sh.set_facecolor(DARK_BG)
        ax_sh.imshow(sh_img)
        ax_sh.axis('off')
        fig_sh.tight_layout(pad=0)
        st.pyplot(fig_sh)
        plt.close()
    elif os.path.exists(shap_bee):
        sh_img = plt.imread(shap_bee)
        fig_sh, ax_sh = plt.subplots(figsize=(8, 5))
        fig_sh.patch.set_facecolor(DARK_BG)
        ax_sh.set_facecolor(DARK_BG)
        ax_sh.imshow(sh_img)
        ax_sh.axis('off')
        fig_sh.tight_layout(pad=0)
        st.pyplot(fig_sh)
        plt.close()
    else:
        st.info("SHAP plots not found. Re-run 04_churn_modeling.py to generate them.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-size: 0.72rem; color: #404660; text-align: center; letter-spacing: 0.04em;'>
        XGBoost · stratified 80/10/10 train/val/test split · AUROC primary metric ·
        SHAP TreeExplainer on 5,000-user validation sample · KKBox WSDM Cup 2018
    </div>
    """, unsafe_allow_html=True)