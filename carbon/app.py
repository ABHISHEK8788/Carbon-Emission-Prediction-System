"""
Streamlit app: Carbon Footprint ML – Data analysis & prediction.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

st.set_page_config(page_title="Carbon Footprint Predictor", page_icon="🌱", layout="wide")

THEME = {
    "bg1": "#0B1020",
    "bg2": "#0F2A3D",
    "card": "rgba(255,255,255,0.08)",
    "card2": "rgba(255,255,255,0.12)",
    "text": "#F9FAFB",          # very light text for dark background
    "muted": "#D1D5DB",         # softer light gray for secondary text
    "accent": "#22C55E",
    "accent2": "#60A5FA",
    "warning": "#FBBF24",
    "danger": "#F43F5E",
}


def inject_css():
    st.markdown(
        f"""
<style>
/* App background */
.stApp {{
  background: radial-gradient(1200px 600px at 20% 10%, rgba(96,165,250,0.25), transparent 55%),
              radial-gradient(900px 500px at 80% 20%, rgba(34,197,94,0.22), transparent 55%),
              linear-gradient(140deg, {THEME['bg1']} 0%, {THEME['bg2']} 100%);
  color: {THEME['text']};
}}

/* Make all text bright on dark background */
.stApp, .stApp p, .stApp span, .stApp li, .stApp label, .stApp div, .stApp td, .stApp th {{
  color: {THEME['text']} !important;
}}

/* Typography */
h1, h2, h3, h4 {{
  letter-spacing: 0.2px;
}}
.muted {{
  color: {THEME['muted']} !important;
}}

/* Cards */
.card {{
  background: {THEME['card']};
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.25);
  backdrop-filter: blur(10px);
}}
.card-strong {{
  background: {THEME['card2']};
  border: 1px solid rgba(255,255,255,0.14);
}}
.badge {{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  color: {THEME['text']};
  background: rgba(34,197,94,0.14);
  border: 1px solid rgba(34,197,94,0.35);
}}

/* Buttons */
div.stButton > button {{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: linear-gradient(135deg, rgba(34,197,94,0.95), rgba(96,165,250,0.95)) !important;
  color: #071019 !important;
  font-weight: 700 !important;
  padding: 0.7rem 1rem !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.25) !important;
}}
div.stButton > button:hover {{
  filter: brightness(1.05);
  transform: translateY(-1px);
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: rgba(255,255,255,0.05);
  border-right: 1px solid rgba(255,255,255,0.10);
}}

/* Input boxes: light background, dark text so typed value is visible */
.stApp input {{
  background-color: #ffffff !important;
  color: #1a1a1a !important;
}}
.stApp input::placeholder {{
  color: #6b7280 !important;
}}
.stApp select {{
  background-color: #ffffff !important;
  color: #1a1a1a !important;
}}
/* Selectbox / dropdown displayed value (Streamlit) */
.stApp [data-baseweb="select"] > div {{
  background-color: #ffffff !important;
  color: #1a1a1a !important;
}}
.stApp [data-testid="stSelectbox"] label {{
  color: {THEME['text']} !important;
}}

/* Dropdown open menu (often rendered in portal, so target body) */
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {{
  background-color: #ffffff !important;
  color: #1a1a1a !important;
  border: 1px solid #e5e7eb !important;
}}
[data-baseweb="popover"] li, [data-baseweb="menu"] li,
[data-baseweb="popover"] [role="option"], [data-baseweb="menu"] [role="option"],
[role="listbox"] li, [role="listbox"] [role="option"] {{
  background-color: #ffffff !important;
  color: #1a1a1a !important;
}}
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover,
[role="listbox"] li:hover {{
  background-color: #f3f4f6 !important;
  color: #1a1a1a !important;
}}
/* Base Web select value display */
[data-baseweb="select"] [aria-selected="true"],
div[data-baseweb="select"] div {{
  background-color: #ffffff !important;
  color: #1a1a1a !important;
}}

/* Any list/popover that looks like dropdown (catch-all for Streamlit) */
ul[role="listbox"], [role="listbox"] ul, div[style*="z-index"] ul {{
  background-color: #ffffff !important;
  color: #1a1a1a !important;
}}
ul[role="listbox"] li, [role="listbox"] ul li {{
  background-color: #ffffff !important;
  color: #1a1a1a !important;
}}
ul[role="listbox"] li:hover, [role="listbox"] ul li:hover {{
  background-color: #e5e7eb !important;
  color: #1a1a1a !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


inject_css()

DATA_PATH = Path(__file__).parent / "eco_optimizer_individual_daily_carbon1.csv"
MODEL_DIR = Path(__file__).parent / "models"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "total_daily_co2_kg"])


def load_model():
    if not (MODEL_DIR / "carbon_model.joblib").exists():
        return None, None, None, None, None
    model = joblib.load(MODEL_DIR / "carbon_model.joblib")
    encoders = joblib.load(MODEL_DIR / "encoders.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.joblib")
    categories = joblib.load(MODEL_DIR / "categories.joblib")
    metrics = None
    metrics_path = MODEL_DIR / "metrics.joblib"
    if metrics_path.exists():
        metrics = joblib.load(metrics_path)
    return model, encoders, feature_cols, categories, metrics


# Load data and model once
df = load_data()
model, encoders, feature_cols, categories, metrics = load_model()

# Simple stateful navigation: landing screen -> prediction screen
if "view" not in st.session_state:
    st.session_state.view = "landing"

if st.session_state.view == "landing":
    # Landing page with hero image and central button
    st.markdown(
        """
<div class="card card-strong">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
    <div>
      <div class="badge">Machine Learning • Sustainability</div>
      <h1 style="margin:10px 0 6px 0;">Carbon Footprint Intelligence</h1>
      <div class="muted" style="font-size: 16px; max-width: 900px;">
        Predict <b>total daily CO₂ (kg)</b> from travel + digital activity and explore insights with an interactive dashboard.
      </div>
    </div>
    <div style="text-align:right;">
      <div class="muted">Fast • Interactive • Visual</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Placeholder hero illustration using Streamlit image (you can replace with a real image path/URL)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.image(
            "https://images.pexels.com/photos/886521/pexels-photo-886521.jpeg",
            caption="Sustainable travel and digital education",
            use_container_width=True,
        )
        st.write("")

        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("<div class='card'><b>Predict</b><div class='muted'>CO₂ estimate from your inputs</div></div>", unsafe_allow_html=True)
        with f2:
            st.markdown("<div class='card'><b>Visualize</b><div class='muted'>KPIs + charts + correlations</div></div>", unsafe_allow_html=True)
        with f3:
            st.markdown("<div class='card'><b>Explain</b><div class='muted'>Model metrics + feature importance</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        cta1, cta2 = st.columns([2, 1])
        with cta1:
            # Centered start button
            if st.button("Start the prediction", use_container_width=True):
                st.session_state.view = "predict"
                st.rerun()
        with cta2:
            if st.button("Open dashboard", use_container_width=True):
                st.session_state.view = "visualize"
                st.rerun()

    # Optional: quick link to explore data in a sidebar
    with st.sidebar:
        st.subheader("Explore")
        if st.button("Visualization dashboard"):
            st.session_state.view = "visualize"
            st.rerun()
        if st.button("Exploratory analysis"):
            st.session_state.view = "explore"
            st.rerun()
        if st.button("View data overview"):
            st.session_state.view = "overview"
            st.rerun()

elif st.session_state.view == "overview":
    st.title("📊 Data Overview")
    st.header("Data Overview")
    display_df = df.head(500).copy()
    display_df["date"] = display_df["date"].astype(str)
    st.dataframe(display_df, width="stretch")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total rows", f"{len(df):,}")
    with c2:
        st.metric("Date range", f"{df['date'].min().date()} → {df['date'].max().date()}")
    with c3:
        st.metric("Avg daily CO₂ (kg)", f"{df['total_daily_co2_kg'].mean():.2f}")
    st.subheader("Column summary")
    st.dataframe(df.describe(), width="stretch")

    if st.button("Back to home"):
        st.session_state.view = "landing"
        st.rerun()

elif st.session_state.view == "explore":
    st.title("📈 Exploratory Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="total_daily_co2_kg", nbins=80, title="Distribution of total_daily_co2_kg")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="transport_mode", y="total_daily_co2_kg", title="CO₂ by transport mode")
        fig.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df, x="role", y="total_daily_co2_kg", title="CO₂ by role")
        fig.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        agg = df.groupby("program")["total_daily_co2_kg"].mean().reset_index()
        fig = px.bar(agg, x="program", y="total_daily_co2_kg", title="Mean CO₂ by program")
        fig.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    # Time series (daily mean)
    daily = df.groupby("date")["total_daily_co2_kg"].mean().reset_index()
    fig = px.line(daily, x="date", y="total_daily_co2_kg", title="Daily average CO₂ over time")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Back to home"):
        st.session_state.view = "landing"
        st.rerun()

elif st.session_state.view == "visualize":
    st.title("📊 Visualization Dashboard")
    st.caption("Interactive summary of carbon emissions with filters, KPIs, and model feature importance.")

    with st.sidebar:
        st.subheader("Filters")
        roles = st.multiselect("Role", options=sorted(df["role"].dropna().unique()), default=[])
        programs = st.multiselect("Program", options=sorted(df["program"].dropna().unique()), default=[])
        transports = st.multiselect(
            "Transport mode", options=sorted(df["transport_mode"].dropna().unique()), default=[]
        )
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    filtered = df.copy()
    if roles:
        filtered = filtered[filtered["role"].isin(roles)]
    if programs:
        filtered = filtered[filtered["program"].isin(programs)]
    if transports:
        filtered = filtered[filtered["transport_mode"].isin(transports)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[(filtered["date"].dt.date >= start) & (filtered["date"].dt.date <= end)]

    tab1, tab2, tab3 = st.tabs(["✨ Summary", "🔎 Deep Dive", "🧠 Model"])

    with tab1:
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Rows", f"{len(filtered):,}")
        with k2:
            st.metric("Avg CO₂ (kg)", f"{filtered['total_daily_co2_kg'].mean():.2f}")
        with k3:
            st.metric("Median CO₂ (kg)", f"{filtered['total_daily_co2_kg'].median():.2f}")
        with k4:
            st.metric("Total CO₂ (kg)", f"{filtered['total_daily_co2_kg'].sum():.0f}")

        # Key charts
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                filtered,
                x="total_daily_co2_kg",
                nbins=80,
                title="Total daily CO₂ distribution",
                color_discrete_sequence=[THEME["accent2"]],
            )
            fig.update_layout(height=360, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(
                filtered,
                x="transport_mode",
                y="total_daily_co2_kg",
                title="CO₂ by transport mode",
                color="transport_mode",
            )
            fig.update_layout(height=360, xaxis_tickangle=-25, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            top_programs = (
                filtered.groupby("program")["total_daily_co2_kg"]
                .mean()
                .sort_values(ascending=False)
                .head(12)
                .reset_index()
            )
            fig = px.bar(
                top_programs,
                x="program",
                y="total_daily_co2_kg",
                title="Top programs by mean CO₂",
                color_discrete_sequence=[THEME["accent"]],
            )
            fig.update_layout(height=360, xaxis_tickangle=-25, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            daily = filtered.groupby("date")["total_daily_co2_kg"].mean().reset_index()
            fig = px.line(
                daily,
                x="date",
                y="total_daily_co2_kg",
                title="Daily average CO₂ over time",
                color_discrete_sequence=[THEME["accent2"]],
            )
            fig.update_layout(height=360, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        # Component contribution (travel vs computer)
        comp = filtered[["travel_co2_kg", "personal_computer_co2_kg"]].sum().reset_index()
        comp.columns = ["component", "co2_kg"]
        comp["component"] = comp["component"].replace(
            {"travel_co2_kg": "Travel CO₂", "personal_computer_co2_kg": "PC CO₂"}
        )
        fig = px.pie(
            comp,
            names="component",
            values="co2_kg",
            title="CO₂ contribution share (Travel vs PC)",
            color_discrete_sequence=[THEME["accent2"], THEME["accent"]],
        )
        fig.update_layout(height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = [
            "distance_km",
            "travel_energy_kwh",
            "travel_co2_kg",
            "personal_computer_kwh",
            "personal_computer_co2_kg",
            "students_trained",
            "lab_energy_kwh",
            "total_daily_co2_kg",
        ]
        corr = filtered[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", title="Correlation heatmap (numeric features)")
        fig.update_layout(height=560, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if metrics:
            st.markdown(
                f"""
<div class="card">
  <div><b>Model:</b> Random Forest Regressor</div>
  <div class="muted">MAE: <b>{metrics['mae']:.4f}</b> • RMSE: <b>{metrics['rmse']:.4f}</b> • R²: <b>{metrics['r2']:.4f}</b></div>
</div>
""",
                unsafe_allow_html=True,
            )

        if model is not None and hasattr(model, "feature_importances_") and feature_cols:
            imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
                "importance", ascending=False
            )
            fig = px.bar(
                imp,
                x="importance",
                y="feature",
                orientation="h",
                title="Model feature importance",
                color_discrete_sequence=[THEME["accent2"]],
            )
            fig.update_layout(height=460, yaxis=dict(autorange="reversed"), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the model first to see feature importance (run `python train_model.py`).")

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Go to prediction"):
            st.session_state.view = "predict"
            st.rerun()
    with nav2:
        if st.button("Go to home"):
            st.session_state.view = "landing"
            st.rerun()

elif st.session_state.view == "predict":
    st.title("🤖 Predict total daily CO₂ (kg)")
    if model is None:
        st.warning("No trained model found. Run `python train_model.py` in the project folder, then refresh.")
        st.code("cd carbon && python train_model.py", language="bash")
    else:
        st.markdown(
            """
<div class="card card-strong">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div>
      <div class="badge">Prediction</div>
      <div class="muted" style="margin-top:6px;">Fill the form below to estimate <b>total_daily_co2_kg</b>.</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if metrics:
            st.markdown(
                f"**Model:** Random Forest regressor  \n"
                f"**Test MAE:** {metrics['mae']:.4f} kg CO₂  \n"
                f"**Test RMSE:** {metrics['rmse']:.4f} kg CO₂  \n"
                f"**Test R²:** {metrics['r2']:.4f}"
            )
        c1, c2, c3 = st.columns(3)
        with c1:
            role = st.selectbox("Role", options=categories["role"])
            program = st.selectbox("Program", options=categories["program"])
            transport_mode = st.selectbox("Transport mode", options=categories["transport_mode"])
        with c2:
            distance_km = st.number_input(
                "Distance (km)",
                min_value=5.0,
                max_value=2000.0,
                value=100.0,
                step=10.0,
                help="Range: 5 – 2000 km (typical in dataset)",
            )
            travel_energy_kwh = st.number_input(
                "Travel energy (kWh)",
                min_value=1.0,
                max_value=1000.0,
                value=50.0,
                step=5.0,
                help="Range: 1 – 1000 kWh (typical in dataset)",
            )
            travel_co2_kg = st.number_input(
                "Travel CO₂ (kg)",
                min_value=0.45,
                max_value=300.0,
                value=15.0,
                step=1.0,
                help="Range: 0.45 – 300 kg (typical in dataset)",
            )
        with c3:
            personal_computer_kwh = st.number_input(
                "PC energy (kWh)",
                min_value=0.24,
                max_value=0.48,
                value=0.36,
                step=0.01,
                help="Range: 0.24 – 0.48 kWh (typical in dataset)",
            )
            personal_computer_co2_kg = st.number_input(
                "PC CO₂ (kg)",
                min_value=0.2,
                max_value=0.39,
                value=0.3,
                step=0.01,
                help="Range: 0.2 – 0.39 kg (typical in dataset)",
            )
            # Only show these for Trainer (otherwise keep them at 0)
            if str(role).strip().lower() == "trainer":
                students_trained = st.number_input("Students trained", min_value=0, value=10, step=1)
                lab_energy_kwh = st.number_input(
                    "Lab energy (kWh)",
                    min_value=0.0,
                    max_value=1.5,
                    value=0.5,
                    step=0.1,
                    help="Range: 0 – 1.5 kWh (typical in dataset)",
                )
            else:
                students_trained = 0
                lab_energy_kwh = 0.0
        month = st.slider("Month", 1, 12, 6)
        day_of_week = st.slider("Day of week (0=Mon)", 0, 6, 2)

        b1, b2, b3 = st.columns(3)
        with b1:
            do_predict = st.button("Predict CO₂", use_container_width=True)
        with b2:
            go_visualize = st.button("Visualize", use_container_width=True)
        with b3:
            go_home = st.button("Go to home", use_container_width=True)

        if go_visualize:
            st.session_state.view = "visualize"
            st.rerun()
        if go_home:
            st.session_state.view = "landing"
            st.rerun()

        if do_predict:
            import numpy as np
            role_enc = encoders["role"].transform([role])[0]
            program_enc = encoders["program"].transform([program])[0]
            transport_enc = encoders["transport_mode"].transform([transport_mode])[0]
            X = pd.DataFrame([{
                "role_encoded": role_enc,
                "program_encoded": program_enc,
                "transport_mode_encoded": transport_enc,
                "distance_km": distance_km,
                "travel_energy_kwh": travel_energy_kwh,
                "travel_co2_kg": travel_co2_kg,
                "personal_computer_kwh": personal_computer_kwh,
                "personal_computer_co2_kg": personal_computer_co2_kg,
                "students_trained": students_trained,
                "lab_energy_kwh": lab_energy_kwh,
                "month": month,
                "day_of_week": day_of_week,
            }])
            X = X[feature_cols]
            pred = model.predict(X)[0]
            st.success(f"**Predicted total daily CO₂:** {pred:.2f} kg")
            st.caption("This is an estimate from the trained Random Forest model.")

            # Input summary + suggestion box
            summary = (
                f"**Role:** {role} · **Program:** {program} · **Transport:** {transport_mode} · "
                f"**Distance:** {distance_km:.0f} km · **Travel CO₂:** {travel_co2_kg:.1f} kg · "
                f"**PC energy:** {personal_computer_kwh:.2f} kWh · **PC CO₂:** {personal_computer_co2_kg:.2f} kg"
            )
            if str(role).strip().lower() == "trainer":
                summary += f" · **Students trained:** {students_trained} · **Lab energy:** {lab_energy_kwh:.1f} kWh"

            suggestions = []
            if transport_mode == "Flight" and distance_km > 500:
                suggestions.append("Consider replacing short/medium flights with train or video calls to cut travel CO₂ significantly.")
            if transport_mode == "Flight":
                suggestions.append("For necessary flights, choose direct routes and economy class to reduce per-passenger emissions.")
            if travel_co2_kg > 100:
                suggestions.append("High travel CO₂: combine trips, carpool, or use public transport where possible.")
            if personal_computer_kwh > 0.4:
                suggestions.append("Lower PC emissions by using power-saving mode and turning off when idle.")
            if str(role).strip().lower() == "trainer" and lab_energy_kwh > 0.5:
                suggestions.append("As a trainer, schedule lab sessions back-to-back to reduce lab energy per session.")
            if pred > 150:
                suggestions.append("Your predicted daily CO₂ is high; small changes in transport or remote work can have a big impact.")
            if distance_km < 50 and transport_mode in ("Cab", "Personal Vehicle", "Auto"):
                suggestions.append("For short distances, walking, cycling, or shared rides can reduce emissions.")
            if not suggestions:
                suggestions.append("Your inputs are within typical ranges. Keep tracking and look for chances to reduce travel or energy use.")

            st.markdown("---")
            with st.expander("Your input summary & suggestions", expanded=True):
                st.markdown(summary)
                st.markdown("**Suggestions:**")
                for i, s in enumerate(suggestions, 1):
                    st.markdown(f"{i}. {s}")
