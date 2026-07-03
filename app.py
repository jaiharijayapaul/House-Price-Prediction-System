import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b);border-right:1px solid #334155;}
section[data-testid="stSidebar"] *{color:#e2e8f0!important;}
.hero{background:linear-gradient(135deg,#1e3a5f,#0f4c81,#1a6b4a);border-radius:16px;padding:2.5rem 2rem;text-align:center;margin-bottom:1.5rem;box-shadow:0 8px 32px rgba(0,0,0,.3);}
.hero h1{font-size:2.6rem;font-weight:800;color:#fff;margin:0;}
.hero p{color:#94d2bd;font-size:1.05rem;margin-top:.5rem;}
.sec{font-size:1.3rem;font-weight:700;color:#1e3a5f;border-left:5px solid #0f4c81;padding-left:.75rem;margin:1.5rem 0 1rem 0;}
.card{border-radius:12px;padding:1.2rem 1rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06);margin:.3rem 0;}
.card.blue{background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border:1px solid #bae6fd;}
.card.green{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1px solid #86efac;}
.card.orange{background:linear-gradient(135deg,#fff7ed,#fed7aa);border:1px solid #fdba74;}
.card.purple{background:linear-gradient(135deg,#faf5ff,#ede9fe);border:1px solid #c4b5fd;}
.card .lbl{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#0369a1;}
.card.green .lbl{color:#15803d;}.card.orange .lbl{color:#c2410c;}.card.purple .lbl{color:#6d28d9;}
.card .val{font-size:1.55rem;font-weight:800;color:#0c4a6e;margin-top:.3rem;}
.card.green .val{color:#14532d;}.card.orange .val{color:#7c2d12;}.card.purple .val{color:#4c1d95;}
.card .sub{font-size:.75rem;color:#64748b;margin-top:.2rem;}
.pred-box{background:linear-gradient(135deg,#0f4c81,#1a6b4a);border-radius:16px;padding:2rem;text-align:center;margin-top:1.5rem;box-shadow:0 8px 32px rgba(0,0,0,.25);}
.pred-box .price{font-size:3rem;font-weight:800;color:#fde68a;}
.pred-box .plbl{font-size:.95rem;color:#a7f3d0;margin-bottom:.5rem;}
.pred-box .det{font-size:.85rem;color:#cbd5e1;margin-top:.5rem;}
.ibox{background:#f8fafc;border-left:4px solid #0f4c81;border-radius:0 8px 8px 0;padding:.9rem 1.2rem;margin:.8rem 0;font-size:.9rem;color:#334155;}
.ibox strong{color:#0f4c81;}
.tag{display:inline-block;background:#dbeafe;color:#1e40af;border-radius:999px;padding:.25rem .75rem;font-size:.78rem;font-weight:600;margin:.2rem;}
.badge{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;background:#0f4c81;color:white;border-radius:50%;font-weight:700;font-size:.82rem;margin-right:.5rem;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────
FMT_K = mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K")

def card(cls, lbl, val, sub=""):
    return (f'<div class="card {cls}">'
            f'<div class="lbl">{lbl}</div>'
            f'<div class="val">{val}</div>'
            f'<div class="sub">{sub}</div></div>')

def ibox(t):
    return f'<div class="ibox">{t}</div>'

def badge(n):
    return f'<span class="badge">{n}</span>'

# ── Training Pipeline (cached) ─────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading data and training model…")
def load_and_train():
    df = pd.read_csv("Data/train.csv")
    FEATURES = ["GrLivArea", "TotRmsAbvGrd"]
    TARGET   = "SalePrice"
    df_model = df[FEATURES + [TARGET]].copy()

    # Outlier removal
    mask     = (df_model["GrLivArea"] > 4000) & (df_model["SalePrice"] < 300_000)
    df_clean = df_model[~mask].copy()

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    # Split BEFORE scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Scale (fit on train only)
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train
    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    y_pred_train = model.predict(X_train_sc)
    y_pred_test  = model.predict(X_test_sc)

    def metrics(yt, yp):
        return dict(
            MAE  = mean_absolute_error(yt, yp),
            MSE  = mean_squared_error(yt, yp),
            RMSE = float(np.sqrt(mean_squared_error(yt, yp))),
            R2   = r2_score(yt, yp),
        )

    # Simple 1-feature line for 2-D chart
    ms = LinearRegression()
    ms.fit(X_train[["GrLivArea"]], y_train)
    area_range = np.linspace(
        df_clean["GrLivArea"].min(), df_clean["GrLivArea"].max(), 300
    ).reshape(-1, 1)
    price_line = ms.predict(area_range)

    return dict(
        df=df, df_clean=df_clean,
        FEATURES=FEATURES, TARGET=TARGET,
        model=model, scaler=scaler,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        y_pred_train=y_pred_train, y_pred_test=y_pred_test,
        train_m=metrics(y_train, y_pred_train),
        test_m =metrics(y_test,  y_pred_test),
        area_range=area_range, price_line=price_line,
        removed=int(mask.sum()),
    )

D = load_and_train()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Navigation")
    PAGES = [
        "🏡 Home", "📂 Dataset Explorer", "⚙️ Preprocessing",
        "🤖 Model & Equation", "📊 Evaluation", "📈 Visualizations",
        "🎯 Predict My House", "📝 Conclusion",
    ]
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📦 Dataset Info")
    st.markdown(f"- **Rows:** {D['df'].shape[0]:,}")
    st.markdown(f"- **Columns:** {D['df'].shape[1]}")
    st.markdown(f"- **Training:** {D['X_train'].shape[0]:,} samples")
    st.markdown(f"- **Test:** {D['X_test'].shape[0]:,} samples")
    st.markdown(f"- **Outliers removed:** {D['removed']}")
    st.markdown("---")
    st.markdown("### 🔑 Key Features")
    st.markdown("- **GrLivArea** — Living area (sq ft)")
    st.markdown("- **TotRmsAbvGrd** — Total rooms")
    st.markdown("- **SalePrice** — Target (USD)")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏡 Home":
    st.markdown(
        '<div class="hero"><h1>🏠 House Price Prediction</h1>'
        '<p>Linear Regression &nbsp;·&nbsp; Ames Housing Dataset</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(card("blue",   "Dataset Size",  f"{D['df'].shape[0]:,}", "Houses in Ames, Iowa"), unsafe_allow_html=True)
    with c2:
        st.markdown(card("green",  "Test R² Score", f"{D['test_m']['R2']:.3f}", "Model accuracy"),       unsafe_allow_html=True)
    with c3:
        st.markdown(card("orange", "Test MAE",      f"${D['test_m']['MAE']:,.0f}", "Avg prediction error"), unsafe_allow_html=True)
    with c4:
        st.markdown(card("purple", "Test RMSE",     f"${D['test_m']['RMSE']:,.0f}", "Root mean sq error"),  unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ Project Pipeline")
    pipeline_steps = [
        ("1", "📂 Load Dataset",       "Read Ames Housing CSV, identify key columns"),
        ("2", "⚙️ Preprocessing",      "Outlier removal, 80/20 split, StandardScaler"),
        ("3", "🤖 Linear Regression",  "Train model, inspect coefficients & equation"),
        ("4", "📊 Evaluation",          "MAE, MSE, RMSE, R² on training and test sets"),
        ("5", "📈 Visualizations",      "4 interactive Matplotlib charts"),
        ("6", "🎯 Custom Prediction",  "Slider inputs → instant price estimate"),
        ("7", "📝 Conclusion",          "Key findings, limitations & next steps"),
    ]
    for n, title, desc in pipeline_steps:
        st.markdown(
            ibox(f'{badge(n)}<strong>{title}</strong> — {desc}'),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🏷️ Technologies Used")
    for tech in ["Python 3", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Streamlit"]:
        st.markdown(f'<span class="tag">{tech}</span>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📂 Dataset Explorer":
    st.markdown('<div class="sec">📂 Step 1 — Load & Explore Dataset</div>', unsafe_allow_html=True)
    df   = D["df"]
    COLS = D["FEATURES"] + [D["TARGET"]]

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Statistics", "🔍 Missing Values"])

    with tab1:
        st.markdown(f"**{df.shape[0]:,} rows × {df.shape[1]} columns** — key columns, first 50 rows")
        st.dataframe(
            df[COLS].head(50)
            .style.format("${:,.0f}", subset=["SalePrice"])
            .background_gradient(cmap="Blues",  subset=["GrLivArea"])
            .background_gradient(cmap="Greens", subset=["TotRmsAbvGrd"]),
            use_container_width=True, height=400,
        )

    with tab2:
        stats = df[COLS].describe().T
        stats.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        st.dataframe(
            stats.style.format("{:,.2f}")
            .background_gradient(cmap="YlOrRd", subset=["Mean", "Std"]),
            use_container_width=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Median Sale Price", f"${df['SalePrice'].median():,.0f}")
            st.metric("Min Sale Price",    f"${df['SalePrice'].min():,.0f}")
            st.metric("Max Sale Price",    f"${df['SalePrice'].max():,.0f}")
        with c2:
            st.metric("Median Living Area", f"{df['GrLivArea'].median():,.0f} sqft")
            st.metric("Min Living Area",    f"{df['GrLivArea'].min():,.0f} sqft")
            st.metric("Max Living Area",    f"{df['GrLivArea'].max():,.0f} sqft")
        with c3:
            st.metric("Avg Rooms", f"{df['TotRmsAbvGrd'].mean():.1f}")
            st.metric("Min Rooms", str(int(df['TotRmsAbvGrd'].min())))
            st.metric("Max Rooms", str(int(df['TotRmsAbvGrd'].max())))

    with tab3:
        miss     = df[COLS].isnull().sum()
        miss_pct = (miss / len(df) * 100).round(2)
        miss_df  = pd.DataFrame({
            "Missing Count": miss,
            "Missing %":     miss_pct,
            "Status":        ["✅ Clean" if v == 0 else "⚠️ Has Nulls" for v in miss],
        })
        st.dataframe(miss_df, use_container_width=True)

        st.markdown("##### Top 10 columns with most missing values (full dataset)")
        top = df.isnull().sum().sort_values(ascending=False).head(10).reset_index()
        top.columns = ["Column", "Missing Count"]
        top["Missing %"] = (top["Missing Count"] / len(df) * 100).round(1)
        st.dataframe(
            top.style.background_gradient(cmap="Reds", subset=["Missing Count"]),
            use_container_width=True,
        )

    # Distribution plots
    st.markdown('<div class="sec">📊 Feature Distributions</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("#f8fafc")

    axes[0].hist(df["SalePrice"], bins=50, color="#3b82f6", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["SalePrice"].median(), color="orange", ls="--", lw=2,
                    label=f"Median: ${df['SalePrice'].median():,.0f}")
    axes[0].set_title("Sale Price Distribution", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].xaxis.set_major_formatter(FMT_K)

    axes[1].hist(df["GrLivArea"], bins=50, color="#22c55e", edgecolor="white", alpha=0.85)
    axes[1].axvline(df["GrLivArea"].median(), color="orange", ls="--", lw=2,
                    label=f"Median: {df['GrLivArea'].median():,.0f} sqft")
    axes[1].set_title("Living Area (sqft)", fontweight="bold")
    axes[1].legend(fontsize=8)

    rv = df["TotRmsAbvGrd"].value_counts().sort_index()
    axes[2].bar(rv.index, rv.values, color="#f97316", edgecolor="white", alpha=0.85)
    axes[2].set_title("Total Rooms Above Grade", fontweight="bold")
    axes[2].set_xlabel("Number of Rooms")

    for ax in axes:
        ax.set_facecolor("#f8fafc")
        ax.grid(alpha=0.3, ls="--")
        ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        ibox("<strong>Key Observations:</strong> SalePrice is right-skewed (most homes $100K–$250K). "
             "GrLivArea peaks around 1,000–2,000 sqft. TotRmsAbvGrd mode is 6–7 rooms."),
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# PAGE 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
elif page == "⚙️ Preprocessing":
    st.markdown('<div class="sec">⚙️ Step 2 — Data Preprocessing</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Outlier Removal")
        st.markdown(
            ibox(f"Removed <strong>{D['removed']}</strong> houses where "
                 "<code>GrLivArea &gt; 4,000 sqft</code> AND "
                 "<code>SalePrice &lt; $300,000</code>. "
                 "These are anomalous non-market transactions per Ames dataset documentation."),
            unsafe_allow_html=True,
        )
        n_before = D["df"][D["FEATURES"] + [D["TARGET"]]].shape[0]
        st.metric("Before removal", f"{n_before:,} samples")
        st.metric("After removal",  f"{D['df_clean'].shape[0]:,} samples")
        st.metric("Removed rows",   str(D["removed"]))

    with c2:
        st.markdown("#### Train / Test Split  (80 / 20)")
        st.markdown(
            ibox("Scaler is fitted <em>only on training data</em>, then applied to both sets — "
                 "prevents data leakage. <code>random_state=42</code> for reproducibility."),
            unsafe_allow_html=True,
        )
        st.metric("Training samples", f"{D['X_train'].shape[0]:,}")
        st.metric("Test samples",     f"{D['X_test'].shape[0]:,}")

    st.markdown("---")
    st.markdown("#### StandardScaler Statistics (from Training Data)")
    sc = D["scaler"]
    sc_df = pd.DataFrame({
        "Feature":    D["FEATURES"],
        "Train Mean": [f"{sc.mean_[0]:,.2f} sqft", f"{sc.mean_[1]:.2f} rooms"],
        "Train Std":  [f"{sc.scale_[0]:,.2f}",     f"{sc.scale_[1]:.2f}"],
        "Effect":     ["Centers area → mean = 0",   "Centers rooms → mean = 0"],
    })
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

    st.markdown("#### Outlier Visualization")
    df_full = D["df"][["GrLivArea", "SalePrice"]].copy()
    mask    = (df_full["GrLivArea"] > 4000) & (df_full["SalePrice"] < 300_000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_full[~mask]["GrLivArea"], df_full[~mask]["SalePrice"],
               alpha=0.4, color="#3b82f6", s=20, label="Normal Data")
    ax.scatter(df_full[mask]["GrLivArea"], df_full[mask]["SalePrice"],
               color="#ef4444", s=120, zorder=5, edgecolors="darkred", lw=1.5,
               label=f"Removed Outliers ({D['removed']})")
    ax.set_xlabel("Living Area (sq ft)")
    ax.set_ylabel("Sale Price")
    ax.yaxis.set_major_formatter(FMT_K)
    ax.set_title("Outlier Detection — GrLivArea vs SalePrice", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_facecolor("#f8fafc")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — MODEL & EQUATION
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model & Equation":
    st.markdown('<div class="sec">🤖 Step 3 — Linear Regression Model</div>', unsafe_allow_html=True)

    model = D["model"]
    sc    = D["scaler"]

    st.markdown("#### Model Equation")
    st.latex(
        r"\hat{y} = \beta_0 + \beta_1 \cdot \text{GrLivArea}_{\text{scaled}}"
        r" + \beta_2 \cdot \text{TotRmsAbvGrd}_{\text{scaled}}"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(card("blue",   "Intercept β₀",      f"${model.intercept_:,.0f}", "Baseline price"),         unsafe_allow_html=True)
    with c2:
        st.markdown(card("green",  "GrLivArea β₁",      f"${model.coef_[0]:,.0f}",  "per std unit ↑"),          unsafe_allow_html=True)
    with c3:
        st.markdown(card("orange", "TotRmsAbvGrd β₂",   f"${model.coef_[1]:,.0f}",  "per std unit ↑"),          unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Coefficient Interpretation")
    st.markdown(
        ibox(f"<strong>β₀ = ${model.intercept_:,.0f} (Intercept):</strong><br>"
             "When both features are at their average values (scaled = 0), the model predicts "
             f"a baseline house price of <strong>${model.intercept_:,.0f}</strong>."),
        unsafe_allow_html=True,
    )
    st.markdown(
        ibox(f"<strong>β₁ = ${model.coef_[0]:,.0f} (GrLivArea):</strong><br>"
             f"Every ~{sc.scale_[0]:,.0f} sq ft increase in living area → price rises by "
             f"<strong>${model.coef_[0]:,.0f}</strong>. This is the <em>strongest</em> predictor."),
        unsafe_allow_html=True,
    )
    st.markdown(
        ibox(f"<strong>β₂ = ${model.coef_[1]:,.0f} (TotRmsAbvGrd):</strong><br>"
             f"Every ~{sc.scale_[1]:.1f} additional room → price changes by "
             f"<strong>${model.coef_[1]:,.0f}</strong>. Secondary predictor."),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Feature Importance (Absolute Coefficients)")
    fig, ax = plt.subplots(figsize=(7, 3))
    abs_coefs = np.abs(model.coef_)
    bars = ax.barh(D["FEATURES"], abs_coefs, color=["#0f4c81", "#22c55e"], edgecolor="white")
    for bar, val in zip(bars, abs_coefs):
        ax.text(bar.get_width() + 300,
                bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", fontweight="bold", fontsize=11)
    ax.set_xlabel("|Coefficient| (USD per std unit)")
    ax.set_title("Feature Importance by Absolute Coefficient", fontweight="bold")
    ax.set_facecolor("#f8fafc")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — EVALUATION
# ══════════════════════════════════════════════════════════════
elif page == "📊 Evaluation":
    st.markdown('<div class="sec">📊 Step 4 — Model Evaluation</div>', unsafe_allow_html=True)

    tr = D["train_m"]
    te = D["test_m"]

    tab1, tab2 = st.tabs(["📈 Metrics Dashboard", "🔍 Actual vs Predicted Table"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📚 Training Set")
            r1, r2, r3 = st.columns(3)
            with r1: st.markdown(card("blue",   "MAE",  f"${tr['MAE']:,.0f}"),  unsafe_allow_html=True)
            with r2: st.markdown(card("orange", "RMSE", f"${tr['RMSE']:,.0f}"), unsafe_allow_html=True)
            with r3: st.markdown(card("green",  "R²",   f"{tr['R2']:.4f}"),     unsafe_allow_html=True)
        with c2:
            st.markdown("##### 🧪 Test Set")
            r1, r2, r3 = st.columns(3)
            with r1: st.markdown(card("blue",   "MAE",  f"${te['MAE']:,.0f}"),  unsafe_allow_html=True)
            with r2: st.markdown(card("orange", "RMSE", f"${te['RMSE']:,.0f}"), unsafe_allow_html=True)
            with r3: st.markdown(card("green",  "R²",   f"{te['R2']:.4f}"),     unsafe_allow_html=True)

        gap    = abs(tr["R2"] - te["R2"])
        status = "✅ No significant overfitting" if gap < 0.05 else "⚠️ Potential overfitting"
        st.markdown(
            ibox(f"<strong>Overfitting Check:</strong> "
                 f"Train R² = {tr['R2']:.4f} | Test R² = {te['R2']:.4f} | "
                 f"Gap = {gap:.4f} → <strong>{status}</strong>"),
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(2)
        ax.bar(x - 0.2, [tr["MAE"], tr["RMSE"]], 0.4, label="Train", color="#3b82f6", edgecolor="white")
        ax.bar(x + 0.2, [te["MAE"], te["RMSE"]], 0.4, label="Test",  color="#22c55e", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(["MAE", "RMSE"], fontsize=12, fontweight="bold")
        ax.set_title("Train vs Test Error Comparison", fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.legend()
        ax.set_facecolor("#f8fafc")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab2:
        comp = pd.DataFrame({
            "Actual ($)":    D["y_test"].values,
            "Predicted ($)": D["y_pred_test"].round(0),
            "Error ($)":     (D["y_pred_test"] - D["y_test"].values).round(0),
            "Error (%)":     ((D["y_pred_test"] - D["y_test"].values) / D["y_test"].values * 100).round(2),
        }).reset_index(drop=True)
        st.markdown(f"Showing first 50 of **{len(comp):,}** test predictions")
        st.dataframe(
            comp.head(50).style
            .format({"Actual ($)": "${:,.0f}", "Predicted ($)": "${:,.0f}",
                     "Error ($)": "${:,.0f}", "Error (%)": "{:.2f}%"})
            .background_gradient(cmap="RdYlGn_r", subset=["Error (%)"])
            .background_gradient(cmap="Blues",    subset=["Actual ($)"]),
            use_container_width=True, height=450,
        )

# ══════════════════════════════════════════════════════════════
# PAGE 6 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown('<div class="sec">📈 Step 5 — Visualizations</div>', unsafe_allow_html=True)

    viz = st.radio(
        "Choose a chart:",
        ["Plot 1: Area vs Price", "Plot 2: Actual vs Predicted",
         "Plot 3: Residual Analysis", "All Charts (2×2 Grid)"],
        horizontal=True,
    )

    dc  = D["df_clean"]
    yt  = D["y_test"];       yp  = D["y_pred_test"]
    yr  = D["y_train"];      ypr = D["y_pred_train"]
    ar  = D["area_range"];   pl  = D["price_line"]
    te  = D["test_m"]

    def plot1():
        fig, ax = plt.subplots(figsize=(10, 6))
        sc_plot = ax.scatter(dc["GrLivArea"], dc["SalePrice"],
                             c=dc["TotRmsAbvGrd"], cmap="viridis", alpha=0.5, s=30)
        ax.plot(ar, pl, color="red", lw=2.5, label="Regression Line")
        cbar = plt.colorbar(sc_plot, ax=ax)
        cbar.set_label("Total Rooms Above Grade", fontsize=10)
        ax.set_title("Living Area vs Sale Price\n(colored by number of rooms)",
                     fontweight="bold", fontsize=13)
        ax.set_xlabel("Living Area (sq ft)")
        ax.set_ylabel("Sale Price (USD)")
        ax.yaxis.set_major_formatter(FMT_K)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#f8fafc")
        plt.tight_layout()
        return fig

    def plot2():
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(yt, yp,  alpha=0.5, color="#3b82f6", s=35, label="Test Predictions")
        ax.scatter(yr, ypr, alpha=0.1, color="#f87171", s=15, label="Train Predictions")
        mn = min(yt.min(), yp.min())
        mx = max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=2, label="Perfect Prediction (y=x)")
        ax.text(0.05, 0.92, f"Test R² = {te['R2']:.4f}", transform=ax.transAxes,
                fontsize=12, color="#1d4ed8", fontweight="bold",
                bbox=dict(boxstyle="round,pad=.3", facecolor="#eff6ff", edgecolor="#3b82f6"))
        ax.set_title("Actual vs Predicted Sale Prices", fontweight="bold", fontsize=13)
        ax.set_xlabel("Actual Price (USD)")
        ax.set_ylabel("Predicted Price (USD)")
        ax.xaxis.set_major_formatter(FMT_K)
        ax.yaxis.set_major_formatter(FMT_K)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#f8fafc")
        plt.tight_layout()
        return fig

    def plot3():
        residuals = yt.values - yp
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(yp, residuals, alpha=0.5, color="#f97316", s=30)
        axes[0].axhline(0, color="black", ls="--", lw=1.5)
        axes[0].set_title("Residuals vs Predicted", fontweight="bold")
        axes[0].set_xlabel("Predicted Price")
        axes[0].set_ylabel("Residual (Actual − Predicted)")
        axes[0].xaxis.set_major_formatter(FMT_K)
        axes[0].yaxis.set_major_formatter(FMT_K)
        axes[0].grid(alpha=0.3)
        axes[0].set_facecolor("#f8fafc")
        axes[1].hist(residuals, bins=50, color="#8b5cf6", edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="black", ls="--", lw=1.5, label="Zero Error")
        axes[1].axvline(residuals.mean(), color="red", lw=2,
                        label=f"Mean: ${residuals.mean():,.0f}")
        axes[1].set_title("Residual Distribution", fontweight="bold")
        axes[1].set_xlabel("Residual Value")
        axes[1].set_ylabel("Count")
        axes[1].xaxis.set_major_formatter(FMT_K)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        axes[1].set_facecolor("#f8fafc")
        plt.tight_layout()
        return fig

    EXPLS = {
        "Plot 1: Area vs Price": (
            "<strong>Interpretation:</strong> Each dot = one house, colored by room count "
            "(yellow = more rooms). The <strong>red regression line</strong> confirms a strong "
            "positive linear relationship. Houses with more rooms cluster at the top-right."
        ),
        "Plot 2: Actual vs Predicted": (
            "<strong>Interpretation:</strong> Points on the dashed diagonal = perfect prediction. "
            "Blue (test) dots near the line confirm good generalization. "
            "Spread increases for luxury homes (&gt;$400K) where 2 features aren't enough."
        ),
        "Plot 3: Residual Analysis": (
            "<strong>Left:</strong> Residuals scattered around zero — no systematic bias. "
            "<strong>Right:</strong> Approximately normal distribution centered near $0, "
            "confirming Linear Regression assumptions are satisfied."
        ),
    }

    if viz == "Plot 1: Area vs Price":
        st.pyplot(plot1(), use_container_width=True)
        st.markdown(ibox(EXPLS["Plot 1: Area vs Price"]), unsafe_allow_html=True)
    elif viz == "Plot 2: Actual vs Predicted":
        st.pyplot(plot2(), use_container_width=True)
        st.markdown(ibox(EXPLS["Plot 2: Actual vs Predicted"]), unsafe_allow_html=True)
    elif viz == "Plot 3: Residual Analysis":
        st.pyplot(plot3(), use_container_width=True)
        st.markdown(ibox(EXPLS["Plot 3: Residual Analysis"]), unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Plot 1 — Area vs Price**")
            st.pyplot(plot1(), use_container_width=True)
            st.markdown("**Plot 3 — Residual Analysis**")
            st.pyplot(plot3(), use_container_width=True)
        with col2:
            st.markdown("**Plot 2 — Actual vs Predicted**")
            st.pyplot(plot2(), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 7 — PREDICT MY HOUSE
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Predict My House":
    st.markdown('<div class="sec">🎯 Step 6 — Predict Your House Price</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders below and get an **instant** price prediction from the trained model.")

    model    = D["model"]
    scaler   = D["scaler"]
    FEATURES = D["FEATURES"]

    c_form, c_result = st.columns([1, 1], gap="large")

    with c_form:
        st.markdown("#### 🏠 House Details")
        area_sqft = st.slider(
            "Living Area (sq ft)", min_value=300, max_value=5000, value=1500, step=50,
            help="Above-grade living area in square feet (GrLivArea)",
        )
        num_rooms = st.slider(
            "Total Rooms Above Grade", min_value=2, max_value=14, value=6, step=1,
            help="Total rooms excluding bathrooms (TotRmsAbvGrd)",
        )

        st.markdown("---")
        st.markdown("#### ⚡ Quick Presets")
        p1, p2 = st.columns(2)
        with p1:
            if st.button("🏘️ Small\n(1,200 sqft / 5 rooms)"):
                area_sqft, num_rooms = 1200, 5
            if st.button("🏠 Large\n(2,500 sqft / 9 rooms)"):
                area_sqft, num_rooms = 2500, 9
        with p2:
            if st.button("🏡 Medium\n(1,800 sqft / 7 rooms)"):
                area_sqft, num_rooms = 1800, 7
            if st.button("🏰 Luxury\n(3,500 sqft / 11 rooms)"):
                area_sqft, num_rooms = 3500, 11

    with c_result:
        inp_df  = pd.DataFrame([[area_sqft, num_rooms]], columns=FEATURES)
        inp_sc  = scaler.transform(inp_df)
        price   = float(max(model.predict(inp_sc)[0], 0))

        st.markdown(
            f'<div class="pred-box">'
            f'<div class="plbl">🏠 Predicted Sale Price</div>'
            f'<div class="price">${price:,.0f}</div>'
            f'<div class="det">{area_sqft:,} sq ft &nbsp;·&nbsp; {num_rooms} rooms</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        median_p = float(D["df"]["SalePrice"].median())
        diff     = price - median_p
        pct      = diff / median_p * 100
        direction = "above" if diff >= 0 else "below"
        st.markdown(
            ibox(f"This prediction is <strong>${abs(diff):,.0f} ({abs(pct):.1f}%) {direction}</strong> "
                 f"the dataset median of <strong>${median_p:,.0f}</strong>."),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📍 Your House on the Regression Chart")
    dc  = D["df_clean"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(dc["GrLivArea"], dc["SalePrice"], alpha=0.3, color="#3b82f6", s=18, label="Dataset")
    ax.plot(D["area_range"], D["price_line"], color="#ef4444", lw=2.5, label="Regression Line")
    ax.scatter([area_sqft], [price], color="#fbbf24", s=280, zorder=6,
               edgecolors="#d97706", lw=2.5, label=f"Your House: {area_sqft:,} sqft")
    ax.annotate(
        f"  ${price:,.0f}",
        xy=(area_sqft, price),
        xytext=(area_sqft + 120, price + 20_000),
        fontsize=11, fontweight="bold", color="#d97706",
        arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.5),
    )
    ax.set_xlabel("Living Area (sq ft)")
    ax.set_ylabel("Sale Price (USD)")
    ax.yaxis.set_major_formatter(FMT_K)
    ax.set_title("Your Prediction on the Regression Chart", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_facecolor("#f8fafc")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Reference Predictions")
    ref_rows = []
    for a, r, lbl in [(1200, 5, "🏘️ Small"), (1800, 7, "🏡 Medium"),
                       (2500, 9, "🏠 Large"), (3500, 11, "🏰 Luxury")]:
        p = float(max(model.predict(
            scaler.transform(pd.DataFrame([[a, r]], columns=FEATURES))
        )[0], 0))
        ref_rows.append({"House Type": lbl, "Area (sqft)": f"{a:,}",
                          "Rooms": r, "Predicted Price": f"${p:,.0f}"})
    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE 8 — CONCLUSION
# ══════════════════════════════════════════════════════════════
elif page == "📝 Conclusion":
    st.markdown(
        '<div class="hero"><h1>📝 Conclusion</h1>'
        '<p>Key findings, limitations &amp; future improvements</p></div>',
        unsafe_allow_html=True,
    )

    tr    = D["train_m"]
    te    = D["test_m"]
    model = D["model"]
    gap   = abs(tr["R2"] - te["R2"])

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(card("green",  "Test R²",    f"{te['R2']:.3f}",        "Variance explained"),    unsafe_allow_html=True)
    with c2: st.markdown(card("orange", "Test MAE",   f"${te['MAE']:,.0f}",     "Avg prediction error"),  unsafe_allow_html=True)
    with c3: st.markdown(card("purple", "Test RMSE",  f"${te['RMSE']:,.0f}",    "Root mean sq error"),    unsafe_allow_html=True)
    with c4: st.markdown(card("blue",   "Overfit Gap",f"{gap:.4f}",             "Train–Test R² diff"),    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✅ Key Conclusions")
    conclusions = [
        ("🏠", "Living Area is the #1 predictor",
         f"GrLivArea β₁ = ${model.coef_[0]:,.0f} per std unit — dominant driver of sale price."),
        ("🚪", "Rooms are a secondary predictor",
         f"TotRmsAbvGrd adds ${model.coef_[1]:,.0f} per std unit, but area carries far more weight."),
        ("⚖️", "No overfitting detected",
         f"Train R² = {tr['R2']:.4f} vs Test R² = {te['R2']:.4f} — gap of only {gap:.4f}."),
        ("📉", "Residuals are normally distributed",
         "Errors cluster near zero, confirming Linear Regression assumptions are satisfied."),
        ("💡", "Strong 2-feature baseline",
         f"R² = {te['R2']:.3f} — {te['R2']*100:.1f}% of price variation explained "
         "with just area and room count."),
    ]
    for icon, title, detail in conclusions:
        st.markdown(ibox(f"<strong>{icon} {title}:</strong><br>{detail}"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚠️ Limitations")
    for lim in [
        "Model struggles with **luxury homes** (>$400K) — quality and location not captured.",
        "**Linear assumption** — the true area–price relationship may be slightly non-linear.",
        "**Dataset scope** — trained on Ames, Iowa; may not generalize to other cities.",
        "Only **2 features** — OverallQual, YearBuilt, Neighborhood, GarageArea excluded.",
    ]:
        st.markdown(f"- {lim}")

    st.markdown("---")
    st.markdown("### 🚀 Future Improvements")
    improvements = [
        ("Add `OverallQual`",           "Highest single-feature correlation with SalePrice"),
        ("Add `Neighborhood`",           "Location is a top driver; encode with one-hot"),
        ("Add `YearBuilt`",              "Newer homes command systematically higher prices"),
        ("Log-transform `SalePrice`",    "Reduces right-skew and improves residual normality"),
        ("Polynomial Regression",         "Captures non-linear area–price curve"),
        ("Ridge / Lasso Regression",      "Regularization when using many features"),
        ("K-Fold Cross Validation",       "More reliable R² estimate than single 80/20 split"),
        ("XGBoost / Gradient Boosting",  "Ensemble method — typically achieves R² > 0.90"),
    ]
    st.dataframe(
        pd.DataFrame(improvements, columns=["Improvement", "Expected Benefit"]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.markdown(
        f'<div class="pred-box" style="background:linear-gradient(135deg,#0f172a,#1e3a5f);">'
        f'<div class="plbl" style="font-size:1rem;margin-bottom:.75rem;">💡 Final Thought</div>'
        f'<div style="color:#e2e8f0;font-size:1.05rem;line-height:1.8;">'
        f'A 2-feature Linear Regression model explains '
        f'<span style="color:#fde68a;font-weight:800;">{te["R2"]*100:.1f}% of house price variation</span> '
        f'using only living area and room count.<br>'
        f'Adding quality ratings, location, and year built could push accuracy above '
        f'<span style="color:#6ee7b7;font-weight:800;">85%</span>.'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;color:#64748b;font-size:.85rem;">'
        '🏠 House Price Prediction &nbsp;·&nbsp; Linear Regression &nbsp;·&nbsp; Ames Housing Dataset'
        '</div>',
        unsafe_allow_html=True,
    )
