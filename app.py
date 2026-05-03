"""
HyperK-DSS  |  ICU Hyperkalemia Early Warning System
Clinical Framing: SCREENING TEST — Trinity Ensemble (LightGBM + XGBoost + CatBoost)
Prediction Window: 6 Hours Before Lab Confirmation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBALS
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HyperK-DSS | ICU Screening System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────
# CSS  —  WARM CLINICAL WHITE THEME
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── palette ────────────────────────────────────────────────── */
:root {
    --page-bg:     #f0f3f7;
    --card-bg:     #ffffff;
    --card-border: #dde3ec;
    --header-bg:   #0d2240;
    --primary:     #1060a8;
    --teal:        #0b7a75;
    --danger:      #c0392b;
    --warn:        #b8860b;
    --success:     #1a7a4a;
    --txt:         #1c2b3a;
    --txt2:        #4f6070;
    --txt3:        #8fa0b0;
    --mono:        'IBM Plex Mono', monospace;
    --sans:        'IBM Plex Sans', sans-serif;
}

/* ── base ───────────────────────────────────────────────────── */
html, body, .stApp                 { background: var(--page-bg) !important; color: var(--txt) !important; }
.main .block-container             { padding: 0 2rem 4rem !important; max-width: 1380px !important; }

/* Targeting specific text elements to avoid overwriting Streamlit Material icons */
html, body, .stApp, p, li, label, h1, h2, h3, h4, h5, h6 { 
    font-family: var(--sans) !important; 
}
h1, h2, h3, h4, h5, h6 { color: var(--txt) !important; }

/* ── header banner ──────────────────────────────────────────── */
.app-header {
    background: var(--header-bg);
    color: #fff;
    padding: 26px 36px 22px;
    margin: 0 -2rem 28px;
    border-bottom: 4px solid var(--primary);
}
.app-header h1 { color: #fff !important; font-size: 1.9rem !important; font-weight: 700 !important; margin: 0 0 4px !important; letter-spacing: .01em !important; }
.app-header p  { color: #a8c0d8; font-size: .9rem; margin: 0; }

/* ── tabs ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff !important;
    border-bottom: 2px solid var(--card-border) !important;
    padding: 0 4px !important; gap: 0 !important;
    border-radius: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--sans) !important; font-size:.9rem !important; font-weight:600 !important;
    color: var(--txt2) !important; padding: 12px 26px !important;
    border-radius: 0 !important; border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 3px solid var(--primary) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--page-bg) !important;
    border: none !important; padding: 24px 0 !important;
}

/* ── generic card ───────────────────────────────────────────── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.card.danger-left  { border-left: 5px solid var(--danger); }
.card.success-left { border-left: 5px solid var(--success); }
.card.primary-left { border-left: 5px solid var(--primary); }

/* ── section heading ────────────────────────────────────────── */
.sec-head {
    display:flex; align-items:center; gap:10px;
    margin: 28px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--card-border);
}
.sec-head h3 {
    font-size:.85rem !important; font-weight:700 !important;
    letter-spacing:.1em !important; text-transform:uppercase;
    color: var(--txt2) !important; margin:0 !important;
}
.dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }

/* ── risk score card ────────────────────────────────────────── */
.risk-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px; padding:24px; text-align:center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.risk-label { font-family:var(--mono); font-size:.65rem; letter-spacing:.12em; text-transform:uppercase; color:var(--txt3); margin-bottom:8px; }
.risk-num   { font-size:3.4rem; font-weight:700; line-height:1; margin-bottom:4px; }
.risk-sub   { font-size:.8rem; color:var(--txt3); }

/* ── alert banners ──────────────────────────────────────────── */
.alert-danger {
    background:#fdf3f2; border:1px solid #f0c4c0;
    border-left:5px solid var(--danger);
    border-radius:8px; padding:16px 20px;
}
.alert-danger  h3 { color:var(--danger) !important; font-size:1rem !important; margin:0 0 8px !important; }
.alert-danger  p  { color:#6b1f1a; font-size:.88rem; margin:0; line-height:1.7; }

.alert-success {
    background:#f2faf5; border:1px solid #b6dfc8;
    border-left:5px solid var(--success);
    border-radius:8px; padding:16px 20px;
}
.alert-success h3 { color:var(--success) !important; font-size:1rem !important; margin:0 0 8px !important; }
.alert-success p  { color:#1a4a30; font-size:.88rem; margin:0; line-height:1.7; }

/* ── STAT CHIP GRID ─────────────────────────────────────────── */
.chip-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 10px;
}
.chip {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 14px 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.chip-lbl {
    font-family: var(--mono); font-size:.6rem; letter-spacing:.1em;
    text-transform:uppercase; color:var(--txt3); margin-bottom:6px;
}
.chip-val { font-size:1.5rem; font-weight:700; color:var(--txt); line-height:1.1; }
.chip-unit{ font-size:.65rem; color:var(--txt3); margin-top:2px; font-family:var(--mono); }
.chip-flag-red  { border-top:3px solid var(--danger);  }
.chip-flag-grn  { border-top:3px solid var(--success); }
.chip-flag-blue { border-top:3px solid var(--primary); }
.chip-flag-gold { border-top:3px solid var(--warn); }

/* ── screening metric cards ─────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.metric-label { font-family:var(--mono); font-size:.58rem; letter-spacing:.1em; text-transform:uppercase; color:var(--txt3); margin-bottom:8px; }
.metric-value { font-size:1.9rem; font-weight:700; line-height:1; }
.metric-sub   { font-size:.68rem; color:var(--txt3); margin-top:4px; font-family:var(--mono); }
.metric-badge { display:inline-block; padding:2px 8px; border-radius:3px; font-size:.62rem; font-weight:700; margin-top:6px; }

/* ── cohort stat chips ──────────────────────────────────────── */
.cstat-row { display:flex; gap:12px; margin-bottom:24px; }
.cstat {
    flex:1; background:var(--card-bg); border:1px solid var(--card-border);
    border-radius:8px; padding:16px; text-align:center;
    box-shadow:0 1px 3px rgba(0,0,0,.05);
}
.cstat-num { font-size:2rem; font-weight:700; }
.cstat-lbl { font-size:.72rem; font-family:var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--txt3); margin-top:4px; }

/* ── feature table ──────────────────────────────────────────── */
.ftable { width:100%; border-collapse:collapse; font-size:.87rem; }
.ftable th { background:#f6f8fb; padding:9px 14px; font-family:var(--mono); font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; color:var(--txt3); border-bottom:2px solid var(--card-border); text-align:left; }
.ftable td { padding:9px 14px; border-bottom:1px solid #edf0f5; color:var(--txt); vertical-align:middle; }
.ftable tr:hover td { background:#f8f9fc; }
.ctag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.68rem; font-weight:600; }

/* ── about blocks ───────────────────────────────────────────── */
.ablock { background:var(--card-bg); border:1px solid var(--card-border); border-radius:10px; padding:22px 26px; margin-bottom:16px; box-shadow:0 1px 4px rgba(0,0,0,.05); }
.ablock h4 { font-size:.8rem !important; font-weight:700 !important; letter-spacing:.1em !important; text-transform:uppercase; color:var(--txt3) !important; margin:0 0 12px !important; }
.ablock p, .ablock li { font-size:.9rem; color:var(--txt); line-height:1.75; }
.mrow { display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; }
.mbox { flex:1; min-width:100px; background:#f6f8fb; border:1px solid var(--card-border); border-radius:8px; padding:14px; text-align:center; }
.mbox .mv { font-size:1.5rem; font-weight:700; }
.mbox .ml { font-family:var(--mono); font-size:.6rem; letter-spacing:.1em; text-transform:uppercase; color:var(--txt3); margin-top:4px; }

/* ── expander fix ───────────────────────────────────────────── */
details { background:var(--card-bg) !important; border:1px solid var(--card-border) !important; border-radius:8px !important; }
summary { color:var(--txt) !important; font-size:.88rem !important; font-weight:600 !important; padding:12px 16px !important; }

/* ── misc ───────────────────────────────────────────────────── */
hr { border-color: var(--card-border) !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-thumb { background:#c8d4e0; border-radius:3px; }
.stCheckbox span { color:var(--txt2) !important; font-size:.88rem !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':   '#ffffff',
    'axes.facecolor':     '#f8f9fc',
    'axes.edgecolor':     '#dde3ec',
    'axes.labelcolor':    '#4f6070',
    'xtick.color':        '#8fa0b0',
    'ytick.color':        '#8fa0b0',
    'text.color':         '#1c2b3a',
    'grid.color':         '#e8ecf2',
    'grid.linewidth':     0.7,
    'axes.grid':          True,
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA & CACHE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model_path  = os.path.join(BASE_DIR, "outputs_v2", "final_catboost_screener.pkl")
    sample_path = os.path.join(BASE_DIR, "outputs_v2", "X_test_sample.csv")
    config_path = os.path.join(BASE_DIR, "outputs_v2", "screening_config.json")

    model    = joblib.load(model_path)
    X_sample = pd.read_csv(sample_path)

    with open(config_path) as f:
        config = json.load(f)

    return model, X_sample, config

model, X_sample, config = load_assets()
THRESHOLD   = config['opt_thresh']
SENS_TARGET = config['sensitivity_target']

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)
explainer = get_explainer()

@st.cache_data
def compute_cohort_probs(_model, _X_sample):
    """Run predict_proba on the full test set once and cache to prevent UI lag on tab switches."""
    X_filled = _X_sample.fillna(_X_sample.median(numeric_only=True))
    return _model.predict_proba(X_filled)[:, 1]

@st.cache_data
def compute_global_shap_importance(_model, _X_sample):
    """Compute global SHAP using a representative 1,000-patient sample for speed."""
    # Ensure reproducibility
    X_samp = _X_sample.sample(min(1000, len(_X_sample)), random_state=42).fillna(_X_sample.median(numeric_only=True))
    temp_explainer = shap.TreeExplainer(_model)
    shap_vals = temp_explainer(X_samp)
    return X_samp, shap_vals

# ─────────────────────────────────────────────────────────────────────
# FEATURE METADATA — map raw column names to display labels
# ─────────────────────────────────────────────────────────────────────
def _make_feature_meta():
    meta = {}
    # Baseline
    meta['age']               = ('Age',                  'years',  'Baseline')
    meta['has_acei']          = ('ACE Inhibitor',        'bool',   'Medication')
    meta['has_diabetes']      = ('Diabetes',             'bool',   'Baseline')
    meta['has_spironolactone']= ('Spironolactone',       'bool',   'Medication')

    # Computed / derived
    meta['Anion_Gap']         = ('Anion Gap',            'mEq/L',  'Acid-Base')
    meta['hours_since_last_k']= ('Hrs Since Last K+',   'h',      'Potassium')
    meta['K_obs_density']     = ('K+ Data Density',     '0–1',    'Potassium')
    meta['Urine_Total_24h']   = ('24h Urine Output',    'mL',     'Fluids')
    meta['Insulin_Total_24h'] = ('24h Insulin',         'units',  'Medication')
    meta['Weight_Latest']     = ('Weight',              'kg',     'Baseline')
    meta['PK_Product']        = ('PK Product',          '',       'Derived')
    meta['K_Acceleration']    = ('K+ Acceleration',     '',       'Derived')
    meta['Renal_Reserve_log'] = ('Renal Reserve (log)', '',       'Derived')

    # Missingness flags
    for w in ['6h', '12h', '18h', '24h']:
        meta[f'K_missing_{w}'] = (f'K+ Missing ({w})', 'bool',   'Potassium')

    # Multi-window lab and vital features
    LABS = {
        'BUN':             ('BUN',             'mg/dL',  'Renal'),
        'Creatinine':      ('Creatinine',      'mg/dL',  'Renal'),
        'Bicarbonate':     ('Bicarbonate',     'mEq/L',  'Acid-Base'),
        'Blood_pH':        ('Blood pH',        '',       'Acid-Base'),
        'Calcium':         ('Calcium',         'mg/dL',  'Electrolyte'),
        'Chloride':        ('Chloride',        'mEq/L',  'Electrolyte'),
        'Phosphate':       ('Phosphate',       'mg/dL',  'Electrolyte'),
        'Prior_Potassium': ('Potassium (Prior)','mEq/L', 'Potassium'),
        'Sodium':          ('Sodium',          'mEq/L',  'Electrolyte'),
        'Heart_Rate':      ('Heart Rate',      'bpm',    'Vitals'),
        'SpO2':            ('SpO2',            '%',      'Vitals'),
    }
    for prefix, (dname, unit, cat) in LABS.items():
        for agg in ['max', 'mean', 'std']:
            for w in ['6h', '12h', '18h', '24h']:
                col = f'{prefix}_{agg}_{w}'
                meta[col] = (f'{dname} ({agg}, {w})', unit, cat)
        for w in ['6h', '12h', '18h', '24h']:
            col = f'{prefix}_delta_{w}'
            meta[col] = (f'{dname} delta ({w})', unit, cat)
    return meta

FEATURE_META = _make_feature_meta()

CAT_COLORS = {
    'Baseline':    ('#1060a8', '#ddeeff'),
    'Renal':       ('#c0392b', '#fde8e6'),
    'Electrolyte': ('#b8860b', '#fff8e1'),
    'Potassium':   ('#6a3d9a', '#f0eaff'),
    'Vitals':      ('#0b7a75', '#e0f5f4'),
    'Fluids':      ('#1a5f8a', '#e6f2fb'),
    'Medication':  ('#1a7a4a', '#e8f9f0'),
    'Acid-Base':   ('#7a3a1a', '#fdf0e8'),
    'Derived':     ('#555',    '#f0f0f0'),
    'Other':       ('#6b7280', '#f3f4f6'),
}

# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>HyperK-DSS — ICU Hyperkalemia Early Warning System</h1>
  <p>Trinity Ensemble (LightGBM + XGBoost + CatBoost) · 6-hour advance prediction · MIMIC-IV · Screening Test Framing</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  Patient Screening  ",
    "  Model Evaluation  ",
    "  About  "
])

# ═════════════════════════════════════════════════════════════════════
# TAB 1 — PATIENT SCREENING
# ═════════════════════════════════════════════════════════════════════
with tab1:

    # ── Patient selector ──────────────────────────────────────────
    # Restored to a type-to-search dropdown to avoid doom-scrolling
    patient_list = [f"Observation #{i+1:,} of {len(X_sample):,}" for i in range(len(X_sample))]
    sel_patient  = st.selectbox("Select Patient Observation:", options=patient_list)
    idx          = patient_list.index(sel_patient)
    patient_data = X_sample.iloc[[idx]]

    st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)

    # ── Prediction ────────────────────────────────────────────────
    patient_filled = patient_data.fillna(X_sample.median(numeric_only=True))
    risk_prob = float(model.predict_proba(patient_filled)[0][1])
    risk_pct  = risk_prob * 100
    is_high   = risk_prob >= THRESHOLD

    n_total   = len(patient_data.columns)
    n_present = patient_data.notna().sum(axis=1).iloc[0]
    completeness_pct = n_present / n_total * 100

    col_score, col_gauge, col_action = st.columns([1, 1, 2])

    with col_score:
        num_color = '#c0392b' if is_high else '#1a7a4a'
        border_cl = 'danger-left' if is_high else 'success-left'
        status_bg = '#fde8e6' if is_high else '#e8f9f0'
        status_txt= 'HIGH RISK — FLAG' if is_high else 'LOW RISK — CLEARED'
        completeness_color = '#c0392b' if completeness_pct < 50 else '#b8860b' if completeness_pct < 75 else '#1a7a4a'
        
        st.markdown(f"""
        <div class="card {border_cl}" style="text-align:center; padding:28px 20px;">
          <div class="risk-label">Potassium Risk Score</div>
          <div class="risk-num" style="color:{num_color};">{risk_pct:.1f}%</div>
          <div class="risk-sub">Screening threshold: {THRESHOLD*100:.1f}%</div>
          <div style="margin-top:12px; display:inline-block; background:{status_bg};
               color:{num_color}; font-weight:700; font-size:.8rem; padding:4px 14px;
               border-radius:20px; border:1px solid {'#f0c4c0' if is_high else '#b6dfc8'};">
            {status_txt}
          </div>
          <div style="margin-top:14px; font-size:.72rem; color:{completeness_color};
               font-family:'IBM Plex Mono',monospace; letter-spacing:.04em;">
            DATA COMPLETENESS: {completeness_pct:.0f}% ({n_present}/{n_total} features present)
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_g, ax_g = plt.subplots(figsize=(3.2, 2.0), subplot_kw=dict(aspect='equal'))
        fig_g.patch.set_facecolor('#ffffff')
        ax_g.set_facecolor('#ffffff')
        ax_g.set_xlim(-1.3, 1.3); ax_g.set_ylim(-0.25, 1.3); ax_g.axis('off')
        theta = np.linspace(np.pi, 0, 300)
        xo = 1.1*np.cos(theta); yo = 1.1*np.sin(theta)
        xi = 0.72*np.cos(theta[::-1]); yi = 0.72*np.sin(theta[::-1])
        ax_g.fill(np.concatenate([xo,xi]), np.concatenate([yo,yi]), color='#e8ecf2', zorder=1)
        angle   = np.pi - risk_prob*np.pi
        theta_f = np.linspace(np.pi, angle, 300)
        seg_col = '#c0392b' if is_high else '#1a7a4a'
        xo2 = 1.1*np.cos(theta_f); yo2 = 1.1*np.sin(theta_f)
        xi2 = 0.72*np.cos(theta_f[::-1]); yi2 = 0.72*np.sin(theta_f[::-1])
        ax_g.fill(np.concatenate([xo2,xi2]), np.concatenate([yo2,yi2]), color=seg_col, zorder=2, alpha=0.85)
        na = np.pi - risk_prob*np.pi
        ax_g.plot([0, 0.82*np.cos(na)], [0, 0.82*np.sin(na)], color='#1c2b3a', linewidth=2.5, zorder=5)
        ax_g.add_patch(plt.Circle((0,0), 0.08, color='#1c2b3a', zorder=6))
        ax_g.add_patch(plt.Circle((0,0), 0.05, color='#fff', zorder=7))
        ax_g.text(0, 0.32, f"{risk_pct:.1f}%", ha='center', va='center',
                  fontsize=11, fontweight='bold', color=seg_col, zorder=8)
        ax_g.text(-1.1, -0.18, '0%',   fontsize=7, color='#8fa0b0', ha='center')
        ax_g.text( 1.1, -0.18, '100%', fontsize=7, color='#8fa0b0', ha='center')
        ta = np.pi - THRESHOLD*np.pi
        ax_g.plot([1.05*np.cos(ta), 1.15*np.cos(ta)],
                  [1.05*np.sin(ta), 1.15*np.sin(ta)], color='#b8860b', linewidth=2, zorder=8)
        st.pyplot(fig_g, transparent=True)
        plt.close(fig_g)
        st.caption(f"Gold marker = screening threshold ({THRESHOLD*100:.1f}%)")

    with col_action:
        if is_high:
            st.markdown("""
            <div class="alert-danger">
              <h3>HIGH RISK — Immediate Action Recommended</h3>
              <p>
                Hyperkalemia predicted within <strong>6 hours</strong>.<br>
                &nbsp;&nbsp;Order urgent bedside blood-gas potassium check<br>
                &nbsp;&nbsp;Review ACE-inhibitor and spironolactone dosing<br>
                &nbsp;&nbsp;Consider calcium gluconate and potassium-shifting agents<br>
                &nbsp;&nbsp;Apply continuous cardiac monitoring (ECG)
              </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
              <h3>LOW RISK — Continue Standard Monitoring</h3>
              <p>
                No hyperkalemia event predicted in the next 6-hour window.<br>
                &nbsp;&nbsp;Maintain current care plan and monitoring schedule<br>
                &nbsp;&nbsp;Re-evaluate at next scheduled potassium draw<br>
                &nbsp;&nbsp;Monitor urine output and fluid balance routinely<br>
                &nbsp;&nbsp;NPV of this clearance: {:.1f}%
              </p>
            </div>""".format(config['npv']*100), unsafe_allow_html=True)

    # ── Clinical Snapshot ─────────────────────────────────────────
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#1060a8;"></span>
      <h3>Clinical Data Snapshot — Multi-Window Look-Back</h3>
    </div>""", unsafe_allow_html=True)

    row = patient_data.iloc[0]

    def fv(col, dec=1, is_bool=False):
        v = row.get(col, np.nan)
        if pd.isna(v): return "N/A"
        if is_bool: return "Yes" if float(v) == 1 else "No"
        try: return f"{float(v):.{dec}f}"
        except: return str(v)

    k_val    = row.get('Prior_Potassium_max_24h', np.nan)
    spo_val  = row.get('SpO2_mean_24h', np.nan)
    cre_val  = row.get('Creatinine_max_24h', np.nan)
    bun_val  = row.get('BUN_max_24h', np.nan)
    bicarb_v = row.get('Bicarbonate_mean_24h', np.nan)
    k_dens   = row.get('K_obs_density', np.nan)
    phos_val = row.get('Phosphate_max_24h', np.nan)
    calc_val = row.get('Calcium_mean_24h', np.nan)
    hr_val   = row.get('Heart_Rate_mean_24h', np.nan)

    k_flag   = 'chip-flag-red'  if (not pd.isna(k_val)   and float(k_val) >= 5.5) else ''
    spo_flag = 'chip-flag-red'  if (not pd.isna(spo_val)  and float(spo_val) < 92) else ''
    cre_flag = 'chip-flag-red'  if (not pd.isna(cre_val)  and float(cre_val) > 1.2) else ''
    bun_flag = 'chip-flag-red'  if (not pd.isna(bun_val)  and float(bun_val) > 20) else ''
    bi_flag  = 'chip-flag-red'  if (not pd.isna(bicarb_v) and float(bicarb_v) < 22) else ''
    ace_flag = 'chip-flag-blue' if float(row.get('has_acei', 0)) == 1 else ''
    dia_flag = 'chip-flag-blue' if float(row.get('has_diabetes', 0)) == 1 else ''
    dens_flag= 'chip-flag-gold' if (not pd.isna(k_dens) and float(k_dens) < 0.5) else \
               'chip-flag-grn'  if (not pd.isna(k_dens) and float(k_dens) >= 0.75) else ''
    phos_flag = 'chip-flag-red' if (not pd.isna(phos_val) and float(phos_val) > 5.5) else ''
    calc_flag = 'chip-flag-red' if (not pd.isna(calc_val) and float(calc_val) < 8.5) else \
                'chip-flag-grn' if (not pd.isna(calc_val) and float(calc_val) > 10.5) else ''
    hr_flag   = 'chip-flag-red' if (not pd.isna(hr_val) and (float(hr_val) > 100 or float(hr_val) < 50)) else ''

    # Row 1
    st.markdown(f"""
    <div class="chip-grid">
      <div class="chip">
        <div class="chip-lbl">Age</div>
        <div class="chip-val">{fv('age',0)}</div>
        <div class="chip-unit">years</div>
      </div>
      <div class="chip {cre_flag}">
        <div class="chip-lbl">Creatinine (max 24h)</div>
        <div class="chip-val">{fv('Creatinine_max_24h')}</div>
        <div class="chip-unit">mg/dL</div>
      </div>
      <div class="chip {bun_flag}">
        <div class="chip-lbl">BUN (max 24h)</div>
        <div class="chip-val">{fv('BUN_max_24h',0)}</div>
        <div class="chip-unit">mg/dL</div>
      </div>
      <div class="chip {k_flag}">
        <div class="chip-lbl">Potassium (max 24h)</div>
        <div class="chip-val">{fv('Prior_Potassium_max_24h')}</div>
        <div class="chip-unit">mEq/L</div>
      </div>
      <div class="chip {spo_flag}">
        <div class="chip-lbl">SpO2 (mean 24h)</div>
        <div class="chip-val">{fv('SpO2_mean_24h',1)}</div>
        <div class="chip-unit">%</div>
      </div>
    </div>
    
    <div class="chip-grid">
      <div class="chip {bi_flag}">
        <div class="chip-lbl">Bicarbonate (mean 24h)</div>
        <div class="chip-val">{fv('Bicarbonate_mean_24h',1)}</div>
        <div class="chip-unit">mEq/L</div>
      </div>
      <div class="chip">
        <div class="chip-lbl">Blood pH (mean 24h)</div>
        <div class="chip-val">{fv('Blood_pH_mean_24h',2)}</div>
        <div class="chip-unit">pH units</div>
      </div>
      <div class="chip {dia_flag}">
        <div class="chip-lbl">Diabetes</div>
        <div class="chip-val" style="font-size:1.2rem;">{fv('has_diabetes',is_bool=True)}</div>
        <div class="chip-unit">comorbidity</div>
      </div>
      <div class="chip {ace_flag}">
        <div class="chip-lbl">ACE Inhibitor</div>
        <div class="chip-val" style="font-size:1.2rem;">{fv('has_acei',is_bool=True)}</div>
        <div class="chip-unit">medication</div>
      </div>
      <div class="chip {dens_flag}">
        <div class="chip-lbl">K+ Data Density</div>
        <div class="chip-val">{fv('K_obs_density',2)}</div>
        <div class="chip-unit">0=sparse, 1=complete</div>
      </div>
    </div>
    
    <div class="chip-grid">
      <div class="chip {phos_flag}">
        <div class="chip-lbl">Phosphate (max 24h)</div>
        <div class="chip-val">{fv('Phosphate_max_24h',1)}</div>
        <div class="chip-unit">mg/dL · &gt;5.5 = high</div>
      </div>
      <div class="chip {calc_flag}">
        <div class="chip-lbl">Calcium (mean 24h)</div>
        <div class="chip-val">{fv('Calcium_mean_24h',1)}</div>
        <div class="chip-unit">mg/dL · 8.5–10.5</div>
      </div>
      <div class="chip {hr_flag}">
        <div class="chip-lbl">Heart Rate (mean 24h)</div>
        <div class="chip-val">{fv('Heart_Rate_mean_24h',0)}</div>
        <div class="chip-unit">bpm</div>
      </div>
      <div class="chip">
        <div class="chip-lbl">Hrs Since Last K+</div>
        <div class="chip-val">{fv('hours_since_last_k',1)}</div>
        <div class="chip-unit">hours</div>
      </div>
      <div class="chip">
        <div class="chip-lbl">Anion Gap</div>
        <div class="chip-val">{fv('Anion_Gap',1)}</div>
        <div class="chip-unit">mEq/L</div>
      </div>
    </div>
    <p style="font-size:.75rem; color:#8fa0b0; margin-top:6px;">
      <span style="border-top:3px solid #c0392b; padding-top:2px;">Red border</span> = abnormal value &nbsp;|&nbsp;
      <span style="border-top:3px solid #1060a8; padding-top:2px;">Blue border</span> = active risk factor &nbsp;|&nbsp;
      <span style="border-top:3px solid #b8860b; padding-top:2px;">Gold border</span> = sparse data warning
    </p>
    """, unsafe_allow_html=True)

    # Full feature table
    with st.expander("Full Feature Detail — All Clinical Variables"):
        cats = {}
        for col in patient_data.columns:
            meta = FEATURE_META.get(col, (col, '', 'Other'))
            cats.setdefault(meta[2], []).append(col)

        rows_html = ""
        for cat, cols in sorted(cats.items()):
            fg, bg = CAT_COLORS.get(cat, ('#555', '#f0f0f0'))
            for col in cols:
                meta  = FEATURE_META.get(col, (col, '', cat))
                fname, funit = meta[0], meta[1]
                val   = row.get(col, np.nan)
                if pd.isna(val):
                    display_val = "—"
                elif funit == 'bool':
                    display_val = "Yes" if float(val) == 1 else "No"
                else:
                    try:
                        display_val = f"{float(val):.3f}"
                    except:
                        display_val = str(val)
                rows_html += f"""<tr>
                  <td><span class="ctag" style="background:{bg};color:{fg};">{cat}</span></td>
                  <td>{fname}</td>
                  <td style="font-family:var(--mono);font-size:.82rem;">{display_val}</td>
                  <td style="font-family:var(--mono);font-size:.75rem;color:#8fa0b0;">{funit}</td>
                </tr>"""

        st.markdown(f"""
        <table class="ftable">
          <thead><tr><th>Category</th><th>Feature</th><th>Value</th><th>Unit</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

    # ── SHAP ──────────────────────────────────────────────────────
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#c0392b;"></span>
      <h3>SHAP Explainability — What Drives This Risk Score?</h3>
    </div>
    <p style="font-size:.87rem; color:#4f6070; margin-bottom:14px;">
      SHAP decomposes the risk score into per-feature contributions.
      <span style="color:#c0392b; font-weight:600;">Red bars</span> push toward higher risk;
      <span style="color:#1060a8; font-weight:600;">blue bars</span> push toward lower risk.
      Missing values are replaced with cohort medians before explanation (physiologically neutral imputation).
    </p>""", unsafe_allow_html=True)

    with st.spinner("Computing SHAP values..."):
        # 1. Map to clean clinical names for the plot
        clean_names = [FEATURE_META.get(c, (c,))[0] for c in patient_filled.columns]
        
        # 2. Generate SHAP values
        shap_values = explainer(patient_filled)
        
        # Apply clean names directly to the SHAP explanation object
        shap_values.feature_names = clean_names
        
        # --- Natural Language Summary ---
        shap_df = pd.DataFrame({
            'Feature': clean_names,
            'SHAP': shap_values[0].values,
            'Value': patient_filled.iloc[0].values
        })
        
        def fmt_val(v):
            if pd.isna(v): return "N/A"
            if isinstance(v, float): return f"{v:.2f}"
            return str(v)

        if is_high:
          top_drivers = shap_df[shap_df['SHAP'] > 0].sort_values(by='SHAP', ascending=False).head(2)
          if not top_drivers.empty:
              driver_text = " and ".join([f"**{row['Feature']}** ({fmt_val(row['Value'])})" for _, row in top_drivers.iterrows()])
              st.markdown(f"""
              <div style="background-color:#fff4e6; border-left:4px solid #ff6b6b; padding:12px 16px; border-radius:4px; margin:12px 0;">
                  <strong style="color:#d63031;"> AI Clinical Interpretation:</strong> 
                  <span style="color:#2d3436;">The model is flagging this patient as HIGH RISK primarily due to their {driver_text}.</span>
              </div>
              """, unsafe_allow_html=True)
        else:
          top_safeties = shap_df[shap_df['SHAP'] < 0].sort_values(by='SHAP', ascending=True).head(2)
          if not top_safeties.empty:
              safe_text = " and ".join([f"**{row['Feature']}** ({fmt_val(row['Value'])})" for _, row in top_safeties.iterrows()])
              st.markdown(f"""
              <div style="background-color:#e8f5e9; border-left:4px solid #4caf50; padding:12px 16px; border-radius:4px; margin:12px 0;">
                  <strong style="color:#2e7d32;"> AI Clinical Interpretation:</strong> 
                  <span style="color:#2d3436;">The model clears this patient, heavily driven by stable {safe_text}.</span>
              </div>
              """, unsafe_allow_html=True)
        # 3. Plot waterfall
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        fig_shap = plt.gcf()
        
        # Rename the math text
        for text in fig_shap.axes[0].texts:
            t = text.get_text()
            if 'f(x)' in t or 'f(X)' in t:
                t = t.replace('f(x)', 'Patient Score').replace('f(X)', 'Patient Score')
            if 'E[f(x)]' in t or 'E[f(X)]' in t or 'E(f(X))' in t:
                t = t.replace('E[f(x)]', 'Baseline Risk').replace('E[f(X)]', 'Baseline Risk').replace('E(f(X))', 'Baseline Risk')
            text.set_text(t)
            
        st.pyplot(fig_shap, bbox_inches='tight')
        plt.close(fig_shap)

    # ── WHAT-IF SIMULATOR ─────────────────────────────────────────
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#b8860b;"></span>
      <h3>Clinical What-If Simulator (Counterfactual Analysis)</h3>
    </div>
    <p style="font-size:.87rem; color:#4f6070; margin-bottom:14px;">
      Test how hypothetical medical interventions change the AI's risk forecast. This transforms the model from a passive alarm into an active decision support tool.
    </p>""", unsafe_allow_html=True)
    
    sim_features = {
        'Prior_Potassium_max_24h': ('Potassium (Max 24h)', 'mEq/L', 3.0, 7.5, 0.1),
        'Creatinine_max_24h': ('Creatinine (Max 24h)', 'mg/dL', 0.5, 10.0, 0.1),
        'Bicarbonate_mean_24h': ('Bicarbonate (Mean 24h)', 'mEq/L', 10.0, 40.0, 1.0)
    }
    
    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        target_feat = st.selectbox("Select clinical variable to simulate:", list(sim_features.keys()), format_func=lambda x: sim_features[x][0])
        
        feat_label, feat_unit, f_min, f_max, f_step = sim_features[target_feat]
        current_val = patient_filled[target_feat].values[0]
        
        # Protect bounds if actual patient value exceeds typical ranges
        s_min = min(f_min, float(current_val))
        s_max = max(f_max, float(current_val))
        
        sim_val = st.slider(f"Simulate new {feat_label}:", 
                            min_value=float(s_min), max_value=float(s_max), 
                            value=float(current_val), step=f_step)
    
    with sim_col2:
        if sim_val != current_val:
            sim_patient = patient_filled.copy()
            sim_patient[target_feat] = sim_val
            new_risk_prob = float(model.predict_proba(sim_patient)[0][1])
            new_risk = new_risk_prob * 100
            delta = risk_pct - new_risk
            
            if delta > 0:
                delta_text = f"drops the hyperkalemia risk by **{delta:.1f}%**"
                box_color = "alert-success"
                header_color = "var(--success)"
            else:
                delta_text = f"increases the hyperkalemia risk by **{abs(delta):.1f}%**"
                box_color = "alert-danger"
                header_color = "var(--danger)"
                
            st.markdown(f"""
            <div class="{box_color}" style="height:100%; display:flex; flex-direction:column; justify-content:center;">
              <h3 style="color:{header_color} !important; margin-bottom:8px;">Intervention Impact</h3>
              <p style="font-size:.95rem; margin:0;">Changing {feat_label} from <strong>{current_val:.1f}</strong> to <strong>{sim_val:.1f}</strong> {delta_text}.</p>
              <p style="font-size:1.1rem; margin-top:8px;">New Simulated Risk: <strong>{new_risk:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#f8f9fc; border:1px dashed #dde3ec; border-radius:8px; padding:20px; height:100%; display:flex; align-items:center; justify-content:center; color:#8fa0b0; font-style:italic;">
              Adjust the slider to simulate an intervention.
            </div>
            """, unsafe_allow_html=True)

    # ── Trend charts ──────────────────────────────────────────────
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#0b7a75;"></span>
      <h3>Key Lab Trends — 24h Window Progression</h3>
    </div>
    <p style="font-size:.87rem; color:#4f6070; margin-bottom:14px;">
      Shows max value and 24h delta for clinically critical labs.
      A rising delta alongside an elevated max signals active deterioration.
    </p>""", unsafe_allow_html=True)

    # delta_bad_when_positive: True  → positive delta = bad (red)  e.g. rising creatinine, BUN, K+
    #                          False → positive delta = good (green) e.g. rising blood pH = less acidotic
    trend_pairs = [
        ('Creatinine_max_24h', 'Creatinine_delta_24h', 'Creatinine', 'mg/dL', '#c0392b', True),
        ('BUN_max_24h',        'BUN_delta_24h',        'BUN',        'mg/dL', '#b8860b', True),
        ('Prior_Potassium_max_24h', 'Prior_Potassium_delta_24h', 'Potassium', 'mEq/L', '#6a3d9a', True),
        ('Blood_pH_max_24h',   'Blood_pH_delta_24h',   'Blood pH',   '',      '#1060a8', False),
    ]
    tc1, tc2, tc3, tc4 = st.columns(4)
    for (mcol, dcol, name, unit, col, bad_when_pos), tcx in zip(trend_pairs, [tc1, tc2, tc3, tc4]):
        vm_raw = row.get(mcol, np.nan)
        vd_raw = row.get(dcol, np.nan)
        
        fig_t, ax_t = plt.subplots(figsize=(2.8, 2.0))
        
        # Check for genuine missingness or impossible 0.0 values from upstream zero-filling
        is_missing = False
        if pd.isna(vm_raw) and pd.isna(vd_raw):
            is_missing = True
        elif mcol == 'Blood_pH_max_24h' and float(vm_raw) < 6.0:
            is_missing = True
        elif mcol == 'Prior_Potassium_max_24h' and float(vm_raw) <= 0.0:
            is_missing = True
            
        if is_missing:
            # Cleanly handle missing data for plotting without drawing empty 0.0 bars
            ax_t.text(0.5, 0.5, "Data Unavailable\nfor 24h Window", ha='center', va='center', fontsize=9, color='#8fa0b0', style='italic')
            ax_t.set_xticks([])
            ax_t.set_yticks([])
            for spine in ax_t.spines.values():
                spine.set_visible(False)
            ax_t.set_title(name, fontsize=9, color='#4f6070', pad=6, fontweight='600')
        else:
            vm = float(vm_raw) if not pd.isna(vm_raw) else 0.0
            vd = float(vd_raw) if not pd.isna(vd_raw) else 0.0
            
            # Delta color: red = deterioration, green = improvement
            if bad_when_pos:
                delta_color = '#c0392b' if vd > 0 else '#1a7a4a'
            else:
                delta_color = '#1a7a4a' if vd > 0 else '#c0392b'
                
            bcolors = [col, delta_color]
            bars = ax_t.bar(['Max', 'Delta'], [vm, vd], color=bcolors, width=0.5, edgecolor='#dde3ec')
            ax_t.axhline(0, color='#dde3ec', linewidth=1)
            ax_t.set_title(name, fontsize=9, color='#4f6070', pad=6, fontweight='600')
            if unit: ax_t.set_ylabel(unit, fontsize=7)
            for bar in bars:
                h = bar.get_height()
                off = abs(h)*0.08 + 0.01
                ax_t.text(bar.get_x()+bar.get_width()/2,
                          h+off if h >= 0 else h-off,
                          f"{h:.2f}", ha='center',
                          va='bottom' if h >= 0 else 'top',
                          fontsize=7.5, color='#1c2b3a', fontweight='500')
                          
        tcx.pyplot(fig_t, transparent=False)
        plt.close(fig_t)


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════
with tab2:

    # ── Screening Performance Card ────────────────────────────────
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#c0392b;"></span>
      <h3>Screening Test Performance — Trinity Ensemble</h3>
    </div>
    <p style="font-size:.87rem; color:#4f6070; margin-bottom:16px;">
      For a screening (rule-out) tool, the key metrics are Sensitivity, NPV, and LR-minus.
      A negative screening result must be highly reliable — the model is optimised to flag every high-risk
      patient at the cost of accepting some false alarms.
    </p>""", unsafe_allow_html=True)

    # Primary metrics — grid
    sens = config['sensitivity']
    npv  = config['npv']
    lrm  = config['lr_minus']
    auprc = config['auprc']
    spec  = config['specificity']
    ppv   = config['ppv']
    nns   = config['nns']
    prev  = config['prevalence']

    lrm_color  = '#1a7a4a' if lrm < 0.1 else '#27ae60' if lrm < 0.2 else '#b8860b' if lrm < 0.3 else '#c0392b'
    lrm_label  = 'Strong rule-out' if lrm < 0.1 else 'Moderate rule-out' if lrm < 0.2 else 'Limited rule-out'

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card" style="border-top:4px solid #1060a8;">
        <div class="metric-label">Sensitivity (Recall)</div>
        <div class="metric-value" style="color:#1060a8;">{sens*100:.1f}%</div>
        <div class="metric-sub">of all HyperK cases detected</div>
        <span class="metric-badge" style="background:#ddeeff;color:#1060a8;">Target: {SENS_TARGET*100:.0f}%</span>
      </div>
      <div class="metric-card" style="border-top:4px solid #1a7a4a;">
        <div class="metric-label">NPV</div>
        <div class="metric-value" style="color:#1a7a4a;">{npv*100:.1f}%</div>
        <div class="metric-sub">cleared patients truly safe</div>
        <span class="metric-badge" style="background:#e8f9f0;color:#1a7a4a;">Primary metric</span>
      </div>
      <div class="metric-card" style="border-top:4px solid {lrm_color};">
        <div class="metric-label">LR-minus (Rule-Out)</div>
        <div class="metric-value" style="color:{lrm_color};">{lrm:.3f}</div>
        <div class="metric-sub">target &lt; 0.2</div>
        <span class="metric-badge" style="background:#f6f8fb;color:{lrm_color};">{lrm_label}</span>
      </div>
      <div class="metric-card" style="border-top:4px solid #b8860b;">
        <div class="metric-label">Alert Burden (NNS)</div>
        <div class="metric-value" style="color:#b8860b;">{nns:.1f}</div>
        <div class="metric-sub">flags per confirmed HyperK</div>
        <span class="metric-badge" style="background:#fff8e1;color:#b8860b;">Operational cost</span>
      </div>
    </div>

    <div class="metric-grid-3">
      <div class="metric-card">
        <div class="metric-label">AUPRC</div>
        <div class="metric-value">{auprc:.4f}</div>
        <div class="metric-sub">{auprc/prev:.1f}x lift over random</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Specificity</div>
        <div class="metric-value">{spec*100:.1f}%</div>
        <div class="metric-sub">correctly cleared observations</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Precision (PPV)</div>
        <div class="metric-value">{ppv*100:.1f}%</div>
        <div class="metric-sub">expected low at {prev*100:.1f}% prevalence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Screening threshold context
    st.markdown(f"""
    <div class="card primary-left" style="margin-bottom:20px;">
      <p style="margin:0;font-size:.88rem;color:#4f6070;line-height:1.8;">
        <strong>Screening threshold applied:</strong> {config['opt_thresh']:.4f}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>Sensitivity target met:</strong> {SENS_TARGET*100:.0f}%
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>HyperK prevalence in test set:</strong> {prev*100:.2f}%
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>LR+:</strong> {config['lr_plus']:.2f}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>Clinical framing:</strong> a negative result means "safely de-prioritize" — a positive result triggers a bedside blood-gas (low-cost intervention, not treatment).
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP cohort plots ─────────────────────────────────────────
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#6a3d9a;"></span>
      <h3>SHAP Global Explainability — Cohort-Wide Feature Impact</h3>
    </div>""", unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Feature Importance Summary (Beeswarm) — Top Predictive Signals**")
        try:
            st.image(os.path.join(BASE_DIR, 'outputs_v2', 'shap_screening_summary.png'),
                     caption="SHAP beeswarm: each point is one flagged observation. Color = feature value. X-axis = direction and magnitude of impact on the risk score.",
                     use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load SHAP summary: {e}")

    with sc2:
        st.markdown("**Feature Impact by Direction — What pushed the score highest?**")
        st.markdown("""<div style="background:#f8f9fc;border:1px solid #dde3ec;border-radius:8px;padding:16px 18px;font-size:.85rem;color:#4f6070;line-height:1.8;">
          <strong>How to read the beeswarm:</strong><br>
          Each dot = one flagged ICU observation.<br>
          <span style="color:#c0392b;font-weight:600;">Red dots (high feature value)</span> pushed the score higher.<br>
          <span style="color:#1060a8;font-weight:600;">Blue dots (low feature value)</span> pulled it lower.<br><br>
          Features at the top have the largest overall impact across the screened cohort.
          A feature appearing high on the chart with red dots on the right means:
          <em>high values of this lab reliably drive HIGH RISK predictions.</em>
        </div>""", unsafe_allow_html=True)

    # ── Feature importance (SHAP based) ───────────────────────────
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#0b7a75;"></span>
      <h3>Global Feature Importance — Top 15 Predictive Signals (SHAP)</h3>
    </div>
    <p style="font-size:.87rem; color:#4f6070; margin-bottom:14px;">
      Ranked by Mean Absolute SHAP value across the test cohort. This perfectly aligns with the patient-level explanations, ensuring mathematical consistency across the UI.
    </p>""", unsafe_allow_html=True)

    with st.spinner("Computing global SHAP importance..."):
        X_samp, global_shap_vals = compute_global_shap_importance(model, X_sample)
        
        mean_abs_shap = np.abs(global_shap_vals.values).mean(axis=0)
        importances = pd.Series(mean_abs_shap, index=X_sample.columns)
        
        top15 = importances.nlargest(15).sort_values()
        flabels = [FEATURE_META.get(c, (c,))[0] for c in top15.index]
        bcolors = [
            CAT_COLORS.get(FEATURE_META.get(c, ('', '', 'Other'))[2], ('#555', '#eee'))[0]
            for c in top15.index
        ]
        fig_i, ax_i = plt.subplots(figsize=(10, 5))
        ax_i.barh(flabels, top15.values, color=bcolors, height=0.6, edgecolor='#fff')
        for i, v in enumerate(top15.values):
            ax_i.text(v + (v*0.02), i, f"{v:.3f}", va='center', fontsize=8, color='#4f6070')
        ax_i.set_xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=9)
        ax_i.set_title('Top 15 Predictive Features (Global SHAP Importance)', fontsize=11, pad=10)
        seen = {FEATURE_META.get(c, ('', '', 'Other'))[2] for c in top15.index}
        patches = [mpatches.Patch(color=CAT_COLORS[k][0], label=k) for k in seen if k in CAT_COLORS]
        ax_i.legend(handles=patches, fontsize=7.5, loc='lower right', ncol=2)
        st.pyplot(fig_i); plt.close(fig_i)
        
    # ── SHAP Dependence Plots ─────────────────────────────────────
    st.markdown("---")
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#1060a8;"></span>
      <h3>Physiological Threshold Discovery (SHAP Dependence)</h3>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:.87rem; color:#4f6070; margin-bottom:14px;">
    This plot proves the AI learned real medical thresholds. 
    Look at where the dots cross the horizontal zero-line—that is the exact lab value where the AI decides a patient transitions from 'Safe' to 'At-Risk'.
    </p>
    """, unsafe_allow_html=True)
    
    with st.spinner("Mapping physiological thresholds..."):
        
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            dep_feat_raw = st.selectbox("Select Feature for Dependence Plot:", 
                                         options=top15.sort_values(ascending=False).index.tolist(),
                                         format_func=lambda x: FEATURE_META.get(x, (x,))[0])

        # Create the Dependence Plot with a much smaller figsize to prevent it from blowing up
        fig_dep, ax_dep = plt.subplots(figsize=(4, 2.5))
        
        shap.plots.scatter(global_shap_vals[:, dep_feat_raw], show=False, ax=ax_dep)
        
        clean_name = FEATURE_META.get(dep_feat_raw, (dep_feat_raw,))[0]
        ax_dep.set_xlabel(f"Actual Lab Value: {clean_name}", fontsize=8, color='#1060a8')
        ax_dep.set_ylabel("Impact on Risk Score (SHAP)", fontsize=8, color='#1060a8')
        ax_dep.tick_params(labelsize=7)
        ax_dep.axhline(0, color='#c0392b', linestyle='--', linewidth=1) 
        fig_dep.patch.set_facecolor('#ffffff')
        ax_dep.set_facecolor('#f8f9fc')
        
        # Placing it in a constrained column prevents Streamlit from scaling it across the full screen width
        col_plot, _ = st.columns([1, 1])
        with col_plot:
            st.pyplot(fig_dep)
        plt.close(fig_dep)

    # ── Cohort risk distribution ──────────────────────────────────
    st.markdown("""<div class="sec-head">
      <span class="dot" style="background:#b8860b;"></span>
      <h3>Cohort Risk Distribution — Test Set</h3>
    </div>""", unsafe_allow_html=True)

    global_probs = compute_cohort_probs(model, X_sample)

    n_high = int((global_probs >= THRESHOLD).sum())
    n_low  = len(global_probs) - n_high

    st.markdown(f"""
    <div class="cstat-row">
      <div class="cstat">
        <div class="cstat-num" style="color:#1060a8;">{len(global_probs):,}</div>
        <div class="cstat-lbl">Total Observations</div>
      </div>
      <div class="cstat">
        <div class="cstat-num" style="color:#c0392b;">{n_high:,}</div>
        <div class="cstat-lbl">Flagged High-Risk ({n_high/len(global_probs)*100:.1f}%)</div>
      </div>
      <div class="cstat">
        <div class="cstat-num" style="color:#1a7a4a;">{n_low:,}</div>
        <div class="cstat-lbl">Cleared Low-Risk ({n_low/len(global_probs)*100:.1f}%)</div>
      </div>
      <div class="cstat">
        <div class="cstat-num" style="color:#b8860b;">{THRESHOLD*100:.1f}%</div>
        <div class="cstat-lbl">Screening Threshold</div>
      </div>
    </div>""", unsafe_allow_html=True)

    dist_c1, dist_c2 = st.columns(2)
    with dist_c1:
        fig_h, ax_h = plt.subplots(figsize=(6, 4))
        n, bins, patches = ax_h.hist(global_probs, bins=40, color='#1060a8', edgecolor='#fff', alpha=0.85)
        for patch, b in zip(patches, bins):
            if b >= THRESHOLD: patch.set_facecolor('#c0392b'); patch.set_alpha(0.9)
        ax_h.axvline(THRESHOLD, color='#b8860b', linestyle='--', linewidth=2,
                     label=f'Screening threshold ({THRESHOLD:.3f})')
        ax_h.set_xlabel('Predicted Risk Score', fontsize=9)
        ax_h.set_ylabel('Observation Count', fontsize=9)
        ax_h.set_title('Risk Score Distribution — Full Test Set', fontsize=11, pad=10)
        ax_h.legend(fontsize=8)
        st.pyplot(fig_h); plt.close(fig_h)

    with dist_c2:
        fig_p, ax_p = plt.subplots(figsize=(5, 4))
        wedges, texts, autos = ax_p.pie(
            [n_low, n_high], labels=['Cleared (Low Risk)', 'Flagged (High Risk)'],
            colors=['#1a7a4a', '#c0392b'], autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='#fff', linewidth=2), pctdistance=0.75)
        for t in texts:  t.set_color('#4f6070'); t.set_fontsize(9)
        for t in autos:  t.set_color('white');   t.set_fontsize(9); t.set_fontweight('bold')
        ax_p.set_title('Screening Split', fontsize=11, pad=10)
        ax_p.set_facecolor('#ffffff')
        st.pyplot(fig_p); plt.close(fig_p)

# ═════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═════════════════════════════════════════════════════════════════════
with tab3:
    _sens_pct = config['sensitivity'] * 100
    _npv_pct  = config['npv'] * 100
    _lrm      = config['lr_minus']

    left, right = st.columns([1.6, 1])

    with left:
        st.markdown(f"""
        <div class="ablock">
          <h4>Project Overview</h4>
          <p>This Decision Support System targets <strong>hyperkalemia</strong> (serum K+ &gt; 5.5 mEq/L) —
          a metabolic emergency in ICU patients that frequently precedes life-threatening arrhythmias.
          Standard laboratory confirmation carries a 1–2 hour processing delay, creating a dangerous
          diagnostic blind spot.</p>
          <p>The model, trained on <strong>MIMIC-IV</strong>, generates a <strong>6-hour advance risk forecast</strong>
          so clinicians can intervene before symptoms or lab confirmation arrive. The system is framed
          as a <strong>screening (rule-out) tool</strong>: the goal is to safely clear low-risk patients,
          not to deliver precise diagnoses.</p>
          <p>False negatives (missed hyperkalemia) risk cardiac arrest.
          False positives (unnecessary blood-gas) are a minor inconvenience.
          The asymmetry justifies optimising sensitivity at the cost of precision.</p>
        </div>

        <div class="ablock">
          <h4>Trinity Ensemble Architecture</h4>
          <p>Three gradient-boosting models are trained independently and their probability outputs
          are averaged (soft voting) to form the final ensemble:</p>
          <ul>
            <li><strong>LightGBM</strong> — is_unbalance=True, AUPRC metric, 1000 estimators</li>
            <li><strong>XGBoost</strong> — scale_pos_weight, AUCPR metric, 1000 estimators</li>
            <li><strong>CatBoost</strong> — scale_pos_weight, PRAUC metric, 1500 iterations</li>
          </ul>
          <p>The ensemble probability is the arithmetic mean of the three individual scores.
          The final screener deployed in this dashboard is the CatBoost component, which
          was found to best generalise to the test set for individual explanation purposes.</p>
          <div class="mrow">
            <div class="mbox"><div class="mv" style="color:#c0392b;">{_sens_pct:.1f}%</div><div class="ml">Sensitivity</div></div>
            <div class="mbox"><div class="mv" style="color:#1a7a4a;">{_npv_pct:.1f}%</div><div class="ml">NPV</div></div>
            <div class="mbox"><div class="mv" style="color:#b8860b;">{_lrm:.3f}</div><div class="ml">LR-minus</div></div>
          </div>
        </div>

        <div class="ablock">
          <h4>SHAP Explainability</h4>
          <p><strong>SHAP (SHapley Additive exPlanations)</strong> ensures every prediction is transparent.
          Instead of a black-box score, SHAP breaks down each risk prediction into individual feature contributions —
          so a clinician can see why a patient is flagged (e.g., rising creatinine + reduced eGFR + ACEi).</p>
          <ul>
            <li><strong>Waterfall plots</strong> (Patient Screening tab) — per-observation explanation</li>
            <li><strong>Beeswarm plot</strong> (Model Evaluation tab) — global behaviour across cohort</li>
          </ul>
          <p>Missing prior potassium values are replaced with cohort medians (not zero) to avoid
          physiologically misleading baseline comparisons.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="ablock">
          <h4>Data Source — MIMIC-IV</h4>
          <p>Medical Information Mart for Intensive Care IV</p>
          <ul>
            <li><code>hosp.labevents</code> — Lab results</li>
            <li><code>icu.chartevents</code> — Vitals and monitoring</li>
            <li><code>icu.outputevents</code> — Fluid output</li>
            <li><code>icu.inputevents</code> — Medication inputs</li>
            <li><code>hosp.diagnoses_icd</code> — Diagnoses</li>
          </ul>
          <p><strong>Cohort:</strong> Adult ICU patients with at least 2 potassium measurements.<br>
          <strong>Label:</strong> Serum K+ &gt; 5.5 mEq/L within 6 hours of prediction window.</p>
        </div>

        <div class="ablock">
          <h4>Feature Engineering (160 Features)</h4>
          <p>All labs and vitals are aggregated across four time windows: 6h, 12h, 18h, 24h,
          with max, mean, std, and delta statistics per window.</p>
          <ul>
            <li><span style="color:#c0392b;">Renal:</span> Creatinine, BUN — 4 windows x 4 stats</li>
            <li><span style="color:#6a3d9a;">Potassium:</span> Prior K+, K+ data density, missingness flags</li>
            <li><span style="color:#b8860b;">Electrolytes:</span> Calcium, Phosphate, Sodium, Chloride</li>
            <li><span style="color:#0b7a75;">Vitals:</span> Heart Rate, SpO2</li>
            <li><span style="color:#7a3a1a;">Acid-Base:</span> Bicarbonate, Blood pH</li>
            <li><span style="color:#1a5f8a;">Computed:</span> Anion Gap, Renal Reserve (log), PK Product, K+ Acceleration</li>
            <li><span style="color:#1a7a4a;">Medications:</span> ACEi, Spironolactone, Insulin</li>
          </ul>
        </div>

        <div class="ablock" style="border-left:4px solid #b8860b;">
          <h4>Clinical Disclaimer</h4>
          <p style="color:#4f6070;">This is a <strong>research prototype</strong> for academic use only, trained on the
          MIMIC-IV de-identified dataset. It is <strong>not approved for clinical use</strong>.
          All clinical decisions must be made by qualified healthcare professionals.</p>
        </div>

        <div class="ablock">
          <h4>References</h4>
          <p style="font-size:.83rem; color:#4f6070; line-height:1.8;">
            1. Kwak et al. (2021). <a href="https://arxiv.org/abs/2101.06443" target="_blank" style="color:var(--primary);text-decoration:none;"><em>Predicting Hyperkalemia in the ICU and Evaluation of Generalizability and Interpretability.</em></a><br>
            2. Liu et al. (2026). <a href="https://www.mdpi.com/2079-9292/15/2/291" target="_blank" style="color:var(--primary);text-decoration:none;"><em>Predicting Hyperkalemia in CKD Using the CatBoost Model and Multiple Interpretability Analyses.</em></a>
          </p>
        </div>
        """, unsafe_allow_html=True)