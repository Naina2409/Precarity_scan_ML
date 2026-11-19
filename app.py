"""PrecarityScan: Streamlit app (workers check their own risk; researchers explore the data and model).
Run: streamlit run app.py"""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import auth
from src.clustering import segment
from src.config import (CLEAN_DATA, FIG_DIR, MODEL_NUM, MODELS_DIR, REPORTS_DIR, RISK_BINS,
                        SUBMISSIONS_FILE, USERS_FILE)
from src.data_cleaner import clean_dataframe
from src.data_generator import CITIES, PLATFORMS
from src.features import engineer
from src.monitoring import drift_report
from src.predict import predict_one
from src.scoring import precarity_index, risk_level
from src.stats_tests import anova, chi_square, welch_ttest

st.set_page_config(page_title="PrecarityScan", page_icon="🛵", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ design tokens
PRIMARY, INK, MUTED = "#0f766e", "#1f2937", "#6b7280"
BAND_COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d9433b"}
BAND_TINT = {"Low": "#e8f6ee", "Medium": "#fff4dc", "High": "#fdeceb"}
SEQ = ["#0f766e", "#2563eb", "#f0a30a", "#d9433b", "#7c3aed", "#0891b2"]
COMPONENT_LABELS = {
    "income_inadequacy": "Income below living wage",
    "income_volatility": "Unstable monthly earnings",
    "benefits_gap": "Missing benefits",
    "platform_dependence": "Depends on one platform",
    "hours_strain": "Very long working week",
    "asset_risk": "Loan tied to work",
}
COMPONENT_HINTS = {
    "income_inadequacy": "Your monthly income is below the assumed living-income benchmark.",
    "income_volatility": "Your earnings swing a lot from month to month, which makes budgeting hard.",
    "benefits_gap": "Health insurance, paid leave and a retirement plan each reduce this risk.",
    "platform_dependence": "Working on a second platform spreads the risk if one cuts your pay or access.",
    "hours_strain": "Working far above a 48-hour week raises burnout and health risk.",
    "asset_risk": "A work-related loan (vehicle etc.) is a fixed cost that continues when income drops.",
}

CSS = f"""
<style>
.block-container {{ padding-top: 4.5rem; max-width: 1200px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ border-right: 1px solid rgba(0,0,0,.06); }}
.hero {{ background: linear-gradient(135deg, {PRIMARY} 0%, #115e59 55%, #134e4a 100%); color: #fff;
        padding: 28px 32px; border-radius: 18px; margin-bottom: 1.2rem; }}
.hero h1 {{ color: #fff; margin: 0 0 6px 0; font-size: 2rem; }}
.hero p {{ color: rgba(255,255,255,.85); margin: 0; font-size: 1.02rem; }}
.pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: rgba(255,255,255,.16); color:#fff; margin: 12px 8px 0 0; }}
.section-label {{ font-size: .74rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                 color: {MUTED}; margin: 0 0 .4rem 0; }}
.band {{ display:inline-block; padding: 6px 18px; border-radius: 999px; font-weight: 700; font-size: 1.05rem; }}
.driver {{ border-left: 4px solid {PRIMARY}; padding: 8px 14px; margin: 8px 0; background: rgba(15,118,110,.06);
          border-radius: 0 10px 10px 0; }}
.driver b {{ color: {INK}; }}
.driver span {{ color: {MUTED}; font-size: .9rem; }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid rgba(0,0,0,.07); border-radius: 14px;
                           padding: 14px 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .85rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; }}
.stButton > button, .stFormSubmitButton > button {{ border-radius: 10px; font-weight: 600; padding: .5rem 1.4rem; }}
.footnote {{ color: {MUTED}; font-size: .82rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def show(fig, height=380):
    """Apply the shared chart style and draw it full-width (works across Streamlit versions)."""
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
                      title=dict(font=dict(size=15)), colorway=SEQ,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2, title=None))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def hero(title, subtitle, pills=()):
    chips = "".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>', unsafe_allow_html=True)


def label(text):
    st.markdown(f'<p class="section-label">{text}</p>', unsafe_allow_html=True)


def gauge(score, band):
    lo, hi = RISK_BINS
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(size=54, color=BAND_COLORS[band])),
        gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor="#cbd5e1"),
                   bar=dict(color=BAND_COLORS[band], thickness=0.28), bgcolor="white", borderwidth=0,
                   steps=[dict(range=[0, lo], color=BAND_TINT["Low"]),
                          dict(range=[lo, hi], color=BAND_TINT["Medium"]),
                          dict(range=[hi, 100], color=BAND_TINT["High"])])))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif"))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def load_data(path, mtime):
    df, _ = clean_dataframe(pd.read_csv(path))
    d = engineer(df)
    d["precarity_index"] = precarity_index(d)
    d["risk_level"] = risk_level(d["precarity_index"]).astype(str)
    return d


@st.cache_data(show_spinner=False)
def cached_segments(path, mtime):
    df, _ = clean_dataframe(pd.read_csv(path))
    return segment(df)


# ------------------------------------------------------------------ login / register
def login_view():
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        hero("🛵 PrecarityScan",
             "Measure how precarious gig work is: income, stability, benefits and dependence.",
             ["Transparent index", "Anonymous by default", "Not financial advice"])
        if not USERS_FILE.exists():
            st.error("No accounts yet. Create the first admin: `python scripts/create_user.py <name> admin`")
        with st.container(border=True):
            t1, t2 = st.tabs(["Log in", "Register (worker)"])
            with t1:
                u = st.text_input("Username", key="lu")
                p = st.text_input("Password", type="password", key="lp")
                if st.button("Log in", type="primary", key="login_btn"):
                    if auth.is_locked(u):
                        st.error("Too many attempts. Try again in a few minutes.")
                    else:
                        role = auth.authenticate(u, p)
                        if role:
                            st.session_state.update(user=u.strip().lower(), role=role)
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
            with t2:
                u = st.text_input("Choose a username", key="ru")
                p = st.text_input("Choose a password", type="password", key="rp")
                st.caption("Min 8 characters with letters and digits. Researcher/admin accounts are created by an administrator.")
                if st.button("Create account", key="register_btn"):
                    try:
                        auth.create_user(u, p, "worker")
                        st.success("Account created. You can log in now.")
                    except ValueError as e:
                        st.error(str(e))
        st.caption("Workers: check your own risk in about a minute. Researchers: explore the data and model. "
                   "Answers are only saved if you tick the consent box.")

# ------------------------------------------------------------------ worker view
def worker_view():
    hero("Check my work situation",
         "Answer a few questions to see how precarious your current gig work is.",
         ["Takes ~1 minute", "Indicator, not advice"])
    st.info("Nothing is stored unless you tick the consent box at the bottom.", icon="🔒")

    with st.form("worker"):
        with st.container(border=True):
            label("1 · About you")
            c1, c2, c3 = st.columns(3)
            city = c1.selectbox("City", list(CITIES))
            platform = c2.selectbox("Main platform", list(PLATFORMS))
            gender = c3.selectbox("Gender (optional)", ["Prefer not to say", "Male", "Female", "Other"],
                                  help="Used only for fairness checks, never for your score.")
            age = c1.number_input("Age", 16, 80, 28)
            exp = c2.number_input("Months on gig work", 0, 360, 24)
            dependents = c3.number_input("People who depend on your income", 0, 10, 1)

        with st.container(border=True):
            label("2 · Your work pattern")
            c1, c2 = st.columns(2)
            hours = c1.slider("Hours per day", 1.0, 18.0, 9.0, 0.5)
            days = c2.slider("Days per week", 1, 7, 6)

        with st.container(border=True):
            label("3 · Your money")
            c1, c2 = st.columns(2)
            income = c1.number_input("Average monthly income (INR)", 0, 300000, 18000, 500)
            savings = c2.number_input("Emergency savings (months of expenses)", 0.0, 24.0, 0.5, 0.1)
            hist = st.text_input("Last 3-6 months' earnings, comma-separated (optional)",
                                 placeholder="17000, 19500, 15000",
                                 help="Adding this lets the score account for how much your income swings.")

        with st.container(border=True):
            label("4 · What protection do you have?")
            b1, b2, b3 = st.columns(3)
            hi = b1.checkbox("Health insurance")
            pl = b2.checkbox("Paid leave")
            rt = b3.checkbox("Retirement plan")
            mp = b1.checkbox("I work on more than one platform")
            ln = b2.checkbox("I have a loan for work (vehicle etc.)")

        consent = st.checkbox("I consent to my anonymous answers being saved for research "
                              "(no name or contact details are kept)")
        go_ = st.form_submit_button("See my result", type="primary")

    if not go_:
        return
    try:
        series = [float(x) for x in hist.split(",") if x.strip()] if hist.strip() else None
        rec = dict(city=city, platform=platform, age=age, months_experience=exp, hours_per_day=hours,
                   days_per_week=days, monthly_income=income, dependents=dependents, savings_months=savings,
                   has_health_insurance=hi, has_paid_leave=pl, has_retirement=rt, multiple_platforms=mp,
                   has_work_loan=ln, gender=None if gender == "Prefer not to say" else gender,
                   monthly_incomes=series)
        out = predict_one(rec)
    except (ValueError, FileNotFoundError) as e:
        st.error(str(e))
        return

    band, score = out["risk_level"], out["precarity_index"]
    st.markdown("## Your result")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        with st.container(border=True):
            label("Precarity index (0-100)")
            gauge(score, band)
            st.markdown(f'<div style="text-align:center"><span class="band" style="background:{BAND_TINT[band]};'
                        f'color:{BAND_COLORS[band]}">{band} risk</span></div>', unsafe_allow_html=True)
            st.write("")
            st.metric("Model: chance of not covering an emergency", f"{out['stress_probability']:.0%}")
    with right:
        with st.container(border=True):
            label("What drives your score")
            comp = (pd.Series(out["all_components"]).rename(index=COMPONENT_LABELS)
                    .rename_axis("component").reset_index(name="points").sort_values("points"))
            fig = px.bar(comp, x="points", y="component", orientation="h", color="points",
                         color_continuous_scale=["#cfe9e6", PRIMARY, "#d9433b"])
            fig.update_layout(coloraxis_showscale=False, xaxis_title="points added to your index", yaxis_title=None)
            show(fig, 300)

    if out["flagged"]:
        st.warning("The model flags a higher-than-usual risk of financial stress. Look at the main drivers below.")
    st.markdown("#### Your biggest drivers")
    inv = {v: k for k, v in COMPONENT_LABELS.items()}
    for name, pts in out["top_drivers"].items():
        key = name if name in COMPONENT_HINTS else inv.get(name, name)
        st.markdown(f'<div class="driver"><b>{COMPONENT_LABELS.get(key, name)}</b> · {pts} points<br>'
                    f'<span>{COMPONENT_HINTS.get(key, "")}</span></div>', unsafe_allow_html=True)
    if not out["history_used"]:
        st.caption("Tip: add your last months' earnings next time to include income volatility in your score.")
    st.markdown('<p class="footnote">This is an indicator based on documented assumptions, not financial advice.</p>',
                unsafe_allow_html=True)

    if consent:
        row = {k: v for k, v in rec.items() if k != "monthly_incomes"}
        pd.DataFrame([row]).to_csv(SUBMISSIONS_FILE, mode="a", header=not SUBMISSIONS_FILE.exists(), index=False)
        st.success("Saved anonymously. Thank you!")


# ------------------------------------------------------------------ researcher views
PAGES = {"📊 Overview": "Overview", "🧪 Statistics": "Statistics", "🧩 Segments": "Segments",
         "🤖 Model": "Model", "🧹 Data & cleaning": "Data", "📡 Monitoring": "Monitoring"}


def page_overview(d):
    hero("Overview", "How precarious is gig work in this dataset?")
    with st.expander("Filters", expanded=False):
        f1, f2 = st.columns(2)
        cities = f1.multiselect("City", sorted(d["city"].dropna().unique()))
        plats = f2.multiselect("Platform", sorted(d["platform"].dropna().unique()))
    if cities: d = d[d["city"].isin(cities)]
    if plats: d = d[d["platform"].isin(plats)]
    if d.empty:
        st.warning("No rows match these filters.")
        return
    
    st.markdown("""
<style>
/* Metric values */
[data-testid="stMetricValue"] {
    color: #1565C0 !important;
    font-size: 35px !important;
    font-weight: bold !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #263238 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


    k = st.columns(5)
    k[0].metric("Workers", f"{len(d):,}")
    k[1].metric("Mean income (INR)", f"{d['monthly_income'].mean():,.0f}")
    k[2].metric("Mean precarity index", f"{d['precarity_index'].mean():.1f}")
    k[3].metric("High-risk share", f"{(d['risk_level'] == 'High').mean():.0%}")
    if "financial_stress" in d:
        k[4].metric("Financial-stress rate", f"{d['financial_stress'].astype(float).mean():.0%}")
    st.write("")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        show(px.box(d, x="platform", y="monthly_income", color="platform", title="Income by platform")
             .update_layout(showlegend=False, xaxis_title=None, yaxis_title="Monthly income (INR)"))
    with c2, st.container(border=True):
        show(px.histogram(d, x="precarity_index", color="risk_level", color_discrete_map=BAND_COLORS, nbins=30,
                          category_orders={"risk_level": ["Low", "Medium", "High"]}, title="Precarity index")
             .update_layout(xaxis_title="Precarity index", yaxis_title="Workers"))
    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        by = d.groupby("city")["precarity_index"].mean().sort_values().reset_index()
        show(px.bar(by, x="precarity_index", y="city", orientation="h", color="precarity_index",
                    color_continuous_scale=["#cfe9e6", PRIMARY, "#d9433b"], title="Mean index by city")
             .update_layout(coloraxis_showscale=False, xaxis_title="Mean index", yaxis_title=None))
    with c4, st.container(border=True):
        mix = pd.crosstab(d["platform"], d["risk_level"], normalize="index").reset_index().melt("platform", value_name="share")
        show(px.bar(mix, x="platform", y="share", color="risk_level", color_discrete_map=BAND_COLORS,
                    category_orders={"risk_level": ["Low", "Medium", "High"]}, title="Risk mix by platform")
             .update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title="Share of workers"))


def page_statistics(d):
    hero("Hypothesis tests", "Pick a question and read the result in plain language.")
    st.markdown("""
<style>
/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #1565C0 !important;
    font-size: 35px !important;
    font-weight: bold !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #263238 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)
    
    d = d.copy()
    d["any_benefit"] = (d["benefits_count"] > 0).astype(int)
    t = st.selectbox("Test", ["Income: multi-platform vs single (Welch t)", "Income across platforms (ANOVA)",
                              "Benefits vs platform (chi-square)"])
    if t.startswith("Income: multi"):
        res = welch_ttest(d, "monthly_income", "multiple_platforms")
        effect = ("Cohen's d", res["cohens_d"], "0.2 small · 0.5 medium · 0.8 large")
    elif "ANOVA" in t:
        res = anova(d, "monthly_income", "platform")
        effect = ("Eta-squared", res["eta_squared"], "0.01 small · 0.06 medium · 0.14 large")
    else:
        res = chi_square(d, "any_benefit", "platform")
        effect = ("Cramér's V", res["cramers_v"], "0.1 small · 0.3 medium · 0.5 large")
    p = res["p_value"]
    sig = p < 0.05
    with st.container(border=True):
        a, b, c = st.columns(3)
        a.metric("p-value", f"{p:.4g}")
        b.metric("Result at 5% level", "Significant" if sig else "Not significant")
        c.metric(effect[0], f"{effect[1]:.3f}", help=effect[2])
        if sig:
            st.success("The difference is unlikely to be due to chance alone. Check the effect size to see if it matters in practice.")
        else:
            st.info("No clear evidence of a difference in this data.")
    st.caption("Correlation is not causation. The data here is synthetic unless you replaced it with real survey data.")
    with st.expander("All test details"):
        st.json(res)


def page_segments():
    hero("Worker segments", "Groups of similar workers found with K-Means.")
    st.markdown("""
<style>
/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #1565C0 !important;
    font-size: 35px !important;
    font-weight: bold !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #263238 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)
    
    seg = cached_segments(str(CLEAN_DATA), CLEAN_DATA.stat().st_mtime)
    k = st.columns(2)
    k[0].metric("Segments (k)", seg["k"])
    k[1].metric("Best silhouette", f"{seg['silhouette']:.2f}", help="Higher means better-separated groups.")
    with st.container(border=True):
        label("Segment profiles (average values)")
        st.dataframe(seg["profile"])
    pc = pd.DataFrame(seg["pca_coords"], columns=["PC1", "PC2"])
    pc["segment"] = seg["labels"].astype(str)
    with st.container(border=True):
        show(px.scatter(pc, x="PC1", y="PC2", color="segment", opacity=0.75, title="PCA projection"), 480)


def page_model():
    hero("Model & evidence", "Registered model versions, figures and the generated report.")
    st.markdown("""
<style>
/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #1565C0 !important;
    font-size: 35px !important;
    font-weight: bold !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #263238 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["Registry", "Figures", "Full report"])
    with t1:
        reg = MODELS_DIR / "registry.json"
        if reg.exists():
            r = json.loads(reg.read_text())
            st.dataframe(pd.DataFrame([{**{k: e[k] for k in ("version", "created_utc", "n_rows", "data_hash")}, **e["metrics"]}
                                       for e in r]))
        else:
            st.warning("No model yet: run `python scripts/run_analysis.py`")
    with t2:
        files = sorted(FIG_DIR.glob("*.png"))
        if not files:
            st.info("No figures yet: run `python scripts/run_analysis.py`")
        for i in range(0, len(files), 2):
            cols = st.columns(2)
            for col, f in zip(cols, files[i:i + 2]):
                with col, st.container(border=True):
                    st.image(str(f), caption=f.stem[3:].replace("_", " ").capitalize())
    with t3:
        rep = REPORTS_DIR / "results.md"
        if rep.exists():
            st.markdown(rep.read_text())
        else:
            st.info("No report yet.")


def page_data(role):
    hero("Data & cleaning", "Upload survey data and get a cleaned, validated file back.")
    st.markdown("""
<style>
/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #1565C0 !important;
    font-size: 35px !important;
    font-weight: bold !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #263238 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)
    
    up = st.file_uploader("CSV file", type="csv")
    if not up:
        st.caption("Drop a survey CSV here. Out-of-range values are set to missing, never silently clipped.")
        return
    raw = pd.read_csv(up)
    clean, rep = clean_dataframe(raw)
    k = st.columns(3)
    k[0].metric("Rows in", f"{len(raw):,}")
    k[1].metric("Rows out", f"{len(clean):,}")
    k[2].metric("Columns", clean.shape[1])
    t1, t2 = st.tabs(["Cleaned preview", "Cleaning report"])
    with t1:
        st.dataframe(clean.head(50))
    with t2:
        st.json(rep)
    st.download_button("Download cleaned CSV", clean.to_csv(index=False), "cleaned.csv", type="primary")
    if role == "admin" and st.button("Replace project dataset with this file"):
        clean.to_csv(CLEAN_DATA, index=False)
        st.cache_data.clear()
        st.success("Dataset replaced. Re-run scripts/run_analysis.py to retrain.")


def page_monitoring():
    hero("Data drift", "Compare new data with what the model was trained on (PSI).")
    st.markdown("""
<style>
/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #1565C0 !important;
    font-size: 35px !important;
    font-weight: bold !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #263238 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)
    
    up = st.file_uploader("New data (CSV)", type="csv", key="drift")
    latest = MODELS_DIR / "latest.json"
    if up and latest.exists():
        ver = json.loads(latest.read_text())["version"]
        ref = pd.read_csv(MODELS_DIR / f"reference_{ver}.csv")
        cur = engineer(clean_dataframe(pd.read_csv(up))[0])
        st.dataframe(drift_report(ref, cur, MODEL_NUM))
        st.caption("PSI < 0.1 stable · 0.1-0.25 moderate · > 0.25 significant drift (consider retraining).")
    else:
        st.caption("Upload new data to compare it with the training reference.")


def researcher_view(role):
    if not CLEAN_DATA.exists():
        st.error("Dataset missing. Run `python scripts/generate_data.py`")
        return
    d = load_data(str(CLEAN_DATA), CLEAN_DATA.stat().st_mtime)
    choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    page = PAGES[choice]
    if page == "Overview": page_overview(d)
    elif page == "Statistics": page_statistics(d)
    elif page == "Segments": page_segments()
    elif page == "Model": page_model()
    elif page == "Data": page_data(role)
    else: page_monitoring()


# ------------------------------------------------------------------ main
if "user" not in st.session_state:
    login_view()
else:
    with st.sidebar:
        st.markdown("### 🛵 PrecarityScan")
        st.caption(f"Signed in as **{st.session_state['user']}** · {st.session_state['role']}")
    role = st.session_state["role"]
    if role == "worker":
        worker_view()
        with st.sidebar:
            st.divider()
    else:
        researcher_view(role)
        st.sidebar.divider()
    if st.sidebar.button("Log out"):
        st.session_state.clear()
        st.rerun()
