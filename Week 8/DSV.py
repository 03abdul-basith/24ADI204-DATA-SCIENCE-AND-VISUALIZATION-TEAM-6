import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Diabetes Health Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent colour meanings throughout the dashboard
NAVY = "#102A43"
TEAL = "#0E9F9A"
CORAL = "#E85D6A"
AMBER = "#F2B544"
GREEN = "#269B72"
BLUE = "#438ACB"
LIGHT_BLUE = "#EAF3F8"
STATUS_COLORS = {"Diabetic": CORAL, "Non-Diabetic": TEAL, "Other / not positive": AMBER}
RISK_SCALE = [[0, "#CDEFEA"], [0.5, "#F9D9A0"], [1, CORAL]]
DATASET_NAME = "kevintan701/diabetes-prediction-datasets"

# =========================================================
# STYLE
# =========================================================
st.markdown(
    f"""
<style>
.stApp {{ background: linear-gradient(180deg,#F3F8FB 0%,#FFFFFF 65%); }}
.block-container {{ max-width: 1450px; padding-top: 1.2rem; padding-bottom: 2rem; }}
h1,h2,h3,h4 {{ color:{NAVY}; }}
section[data-testid="stSidebar"] {{ background:#EAF3F4; border-right:1px solid #D7E5E8; }}
section[data-testid="stSidebar"] * {{ color:{NAVY}; }}
div[data-testid="stMetric"] {{ background:#FFFFFF; border:1px solid #E1EAF0; border-radius:13px; padding:14px; }}
.hero {{ background:linear-gradient(120deg,{NAVY},#145477 58%,{TEAL}); padding:28px 32px; border-radius:18px; margin-bottom:18px; box-shadow:0 8px 22px rgba(16,42,67,.14); }}
.hero h1 {{ color:#FFFFFF !important; margin:0 0 7px 0; font-size:2.05rem; }}
.hero p {{ color:#E2F2F4 !important; margin:0; font-size:1rem; }}
.chip {{ display:inline-block; background:rgba(255,255,255,.16); color:white !important; padding:4px 11px; border-radius:18px; margin:12px 7px 0 0; font-size:.8rem; }}
.kpi {{ background:white; border:1px solid #E3EBF2; border-left:6px solid var(--accent); border-radius:13px; padding:15px 16px; min-height:105px; box-shadow:0 3px 10px rgba(11,37,69,.04); }}
.kpi-label {{ color:#627487; font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; }}
.kpi-value {{ color:{NAVY}; font-size:1.8rem; font-weight:750; line-height:1.3; }}
.kpi-sub {{ color:#7A8896; font-size:.78rem; }}
.info-note {{ background:#EEF7FF; border-left:5px solid {BLUE}; border-radius:9px; padding:10px 14px; margin:8px 0 14px; color:#1D3B5C; }}
.insight {{ background:white; border:1px solid #E5EBF1; border-left:5px solid {AMBER}; border-radius:10px; padding:11px 14px; margin:7px 0; color:#263D53; }}
[data-testid="stDataFrame"] {{ border:1px solid #E2EAF0; border-radius:10px; }}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SMALL REUSABLE FUNCTIONS
# =========================================================
def normalize_name(value):
    name = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", name).strip("_")


def kpi_card(label, value, detail, accent):
    st.markdown(
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def info_note(message):
    st.markdown(f'<div class="info-note">💡 {message}</div>', unsafe_allow_html=True)


def format_name(column):
    return str(column).replace("_", " ").title()


def chart_style(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=58, b=18),
        font=dict(family="Arial, sans-serif", size=13, color="#263D53"),
        title=dict(font=dict(size=16, color=NAVY)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def show_chart(fig, height=380):
    st.plotly_chart(chart_style(fig, height), use_container_width=True)


def detect_column(columns, aliases):
    """Find a column by exact normalized alias first, then by a safe name hint."""
    cols = list(columns)
    for alias in aliases:
        if alias in cols:
            return alias
    for col in cols:
        tokens = col.split("_")
        for hint in aliases:
            # Match short names such as "age" as whole words, so "average"
            # is not accidentally mistaken for an age column.
            if ("_" in hint and hint in col) or ("_" not in hint and hint in tokens):
                return col
    return None


def prevalence_table(data, group_column):
    temp = data[[group_column, "_diabetic"]].dropna().copy()
    if temp.empty:
        return pd.DataFrame(columns=[group_column, "Diabetes Rate (%)", "Cases", "Records"])
    result = (
        temp.groupby(group_column, observed=True)["_diabetic"]
        .agg(rate="mean", cases="sum", records="count")
        .reset_index()
    )
    result["Diabetes Rate (%)"] = result["rate"] * 100
    return result.drop(columns="rate").rename(columns={"cases": "Cases", "records": "Records"})


def prevalence_chart(data, group_column, title, overall_rate=None, min_group_size=10, height=360):
    result = prevalence_table(data, group_column)
    result = result[result["Records"] >= min_group_size].copy()
    if result.empty:
        st.info(f"Not enough records in the groups for {title.lower()}.")
        return result
    result[group_column] = result[group_column].astype(str)
    fig = px.bar(
        result,
        x=group_column,
        y="Diabetes Rate (%)",
        color="Diabetes Rate (%)",
        color_continuous_scale=RISK_SCALE,
        text=result["Diabetes Rate (%)"].map(lambda x: f"{x:.1f}%"),
        hover_data={"Cases": True, "Records": True},
        title=title,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="Positive-label rate (%)", rangemode="tozero")
    fig.update_xaxes(title="")
    if overall_rate is not None:
        fig.add_hline(
            y=overall_rate,
            line_dash="dash",
            line_color=NAVY,
            annotation_text=f"Overall {overall_rate:.1f}%",
            annotation_position="top left",
        )
    show_chart(fig, height)
    return result


def safe_mean(data, column, digits=1):
    if not column or column not in data.columns:
        return "N/A"
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    return f"{values.mean():.{digits}f}" if not values.empty else "N/A"

# =========================================================
# LOAD DATA FROM KAGGLE
# =========================================================
@st.cache_data(show_spinner="Connecting to Kaggle and downloading the dataset...")
def download_dataset_files():
    import kagglehub
    folder = Path(kagglehub.dataset_download(DATASET_NAME))
    csv_files = sorted(folder.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("The Kaggle dataset download did not contain a CSV file.")
    return [str(file) for file in csv_files]


@st.cache_data(show_spinner="Reading the selected CSV file...")
def read_csv_file(file_path):
    return pd.read_csv(file_path, low_memory=False)


st.sidebar.title("⚙️ Dashboard Controls")

try:
    csv_files = download_dataset_files()
except Exception as error:
    st.error("The Kaggle dataset could not be downloaded.")
    st.write("Check your internet connection and Kaggle authentication, then restart the app.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()

# Prefer a filename that looks like the main diabetes dataset.
default_file_index = next(
    (i for i, file in enumerate(csv_files) if "diabetes" in Path(file).name.lower()),
    0,
)
selected_file = st.sidebar.selectbox(
    "Dataset file",
    csv_files,
    index=default_file_index,
    format_func=lambda file: Path(file).name,
)

try:
    raw_df = read_csv_file(selected_file)
except Exception as error:
    st.error(f"Could not read {Path(selected_file).name}.")
    st.code(str(error))
    st.stop()

if raw_df.empty:
    st.error("The selected CSV file is empty.")
    st.stop()

# =========================================================
# DATA PREPARATION
# =========================================================
df = raw_df.copy()
original_columns = df.columns.tolist()
df.columns = [normalize_name(col) for col in df.columns]
df = df.loc[:, ~df.columns.duplicated()].copy()

# Convert columns that are almost entirely numeric.
for column in df.select_dtypes(include="object").columns:
    converted = pd.to_numeric(df[column], errors="coerce")
    non_missing = df[column].notna().sum()
    if non_missing and converted.notna().sum() / non_missing >= 0.90:
        df[column] = converted

df = df.replace([np.inf, -np.inf], np.nan)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
all_columns = df.columns.tolist()

# =========================================================
# COLUMN MAPPING - USER CAN OVERRIDE AUTOMATIC DETECTION
# =========================================================
known_target = detect_column(
    all_columns,
    ["diabetes", "diabetes_status", "diabetes_binary", "diabetes_012", "outcome", "target", "class", "diagnosis"],
)

st.sidebar.subheader("🧭 Column Mapping")
target_choices = ["-- Select target column --"] + all_columns
target_default = target_choices.index(known_target) if known_target in all_columns else 0
target_column = st.sidebar.selectbox("Diabetes outcome / target", target_choices, index=target_default)

if target_column == "-- Select target column --":
    st.warning("Choose the column that contains the diabetes outcome in the sidebar.")
    st.write("Available columns:", ", ".join(all_columns))
    st.dataframe(df.head(8), use_container_width=True)
    st.stop()

# Explicitly choose which target values represent diabetes. Do not silently assume
# that the rarest category is diabetic.
target_values = df[target_column].dropna().unique().tolist()
if len(target_values) < 2:
    st.error("The selected target column must contain at least two different values.")
    st.stop()

positive_words = {"1", "1.0", "yes", "true", "positive", "pos", "diabetes", "diabetic", "present", "y"}
default_positive = []
for value in target_values:
    text_value = str(value).strip().lower()
    if text_value in positive_words:
        default_positive.append(value)
if not default_positive and target_column == "diabetes_012" and 2 in target_values:
    default_positive = [2]
if not default_positive and target_column in {"diabetes", "diabetes_binary", "outcome", "target", "class"} and 1 in target_values:
    default_positive = [1]

positive_values = st.sidebar.multiselect(
    "Which target value(s) mean diabetes?",
    options=target_values,
    default=default_positive,
    help="For a 0/1 target, choose 1. For diabetes_012, usually choose 2. Check your dataset documentation.",
)

if not positive_values:
    st.warning("Select at least one positive diabetes value in the sidebar. This prevents incorrect case counts.")
    st.stop()

# Missing target values are excluded rather than incorrectly counted as non-diabetic.
valid_target = df[target_column].notna()
df = df.loc[valid_target].copy()
df["_diabetic"] = df[target_column].isin(positive_values).astype(int)

is_binary_target = df[target_column].nunique(dropna=True) <= 2
negative_label = "Non-Diabetic" if is_binary_target else "Other / not positive"
df["_status"] = np.where(df["_diabetic"].eq(1), "Diabetic", negative_label)

if df["_diabetic"].sum() == 0:
    st.error("No positive cases were found. Recheck the selected target values in the sidebar.")
    st.stop()
if df["_diabetic"].sum() == len(df):
    st.warning("Every selected record is marked positive. Please verify the target mapping.")

# Suggested column mapping. Manual selectors solve cases where dataset names differ.
def mapped_selector(label, aliases, numeric_only=False):
    pool = numeric_columns if numeric_only else all_columns
    pool = [c for c in pool if c != target_column and not c.startswith("_")]
    options = ["Not available"] + pool
    guess = detect_column(pool, aliases)
    index = options.index(guess) if guess in options else 0
    chosen = st.sidebar.selectbox(label, options, index=index)
    return None if chosen == "Not available" else chosen

age_col = mapped_selector("Age column", ["age", "patient_age"], True)
bmi_col = mapped_selector("BMI column", ["bmi", "body_mass_index"], True)
glucose_col = mapped_selector("Blood glucose column", ["blood_glucose_level", "blood_glucose", "glucose"], True)
hba1c_col = mapped_selector("HbA1c column", ["hba1c_level", "hba1c", "hb_a1c"], True)
gender_col = mapped_selector("Gender column", ["gender", "sex"])
smoking_col = mapped_selector("Smoking column", ["smoking_history", "smoker", "smoking"])

# =========================================================
# DERIVED HEALTH GROUPS
# =========================================================
AGE_LABELS = ["Under 18", "18–29", "30–44", "45–59", "60–74", "75+"]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese", "Severely obese"]
A1C_LABELS = ["Below 5.7", "5.7–6.4", "6.5 or above"]
GLUCOSE_LABELS = ["Below 140", "140–199", "200 or above"]

if age_col:
    df["_age_group"] = pd.cut(
        pd.to_numeric(df[age_col], errors="coerce"),
        bins=[-np.inf, 18, 30, 45, 60, 75, np.inf],
        labels=AGE_LABELS,
        right=False,
    )
if bmi_col:
    df["_bmi_group"] = pd.cut(
        pd.to_numeric(df[bmi_col], errors="coerce"),
        bins=[-np.inf, 18.5, 25, 30, 40, np.inf],
        labels=BMI_LABELS,
        right=False,
    )
if hba1c_col:
    df["_hba1c_group"] = pd.cut(
        pd.to_numeric(df[hba1c_col], errors="coerce"),
        bins=[-np.inf, 5.7, 6.5, np.inf],
        labels=A1C_LABELS,
        right=False,
    )
if glucose_col:
    df["_glucose_group"] = pd.cut(
        pd.to_numeric(df[glucose_col], errors="coerce"),
        bins=[-np.inf, 140, 200, np.inf],
        labels=GLUCOSE_LABELS,
        right=False,
    )

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("🔎 Patient Filters")
filtered = df.copy()

if gender_col:
    options = sorted(filtered[gender_col].dropna().astype(str).unique().tolist())
    selected = st.sidebar.multiselect("Gender", options, default=options)
    filtered = filtered[filtered[gender_col].astype(str).isin(selected)]

if age_col and filtered[age_col].notna().any():
    low, high = float(filtered[age_col].min()), float(filtered[age_col].max())
    if low < high:
        age_range = st.sidebar.slider("Age range", min_value=low, max_value=high, value=(low, high))
        filtered = filtered[filtered[age_col].between(*age_range)]

if bmi_col and filtered[bmi_col].notna().any():
    low, high = float(filtered[bmi_col].min()), float(filtered[bmi_col].max())
    if low < high:
        bmi_range = st.sidebar.slider("BMI range", min_value=low, max_value=high, value=(low, high))
        filtered = filtered[filtered[bmi_col].between(*bmi_range)]

if smoking_col:
    options = sorted(filtered[smoking_col].dropna().astype(str).unique().tolist())
    selected = st.sidebar.multiselect("Smoking history", options, default=options)
    filtered = filtered[filtered[smoking_col].astype(str).isin(selected)]

for flag in ["hypertension", "heart_disease"]:
    if flag in filtered.columns and set(filtered[flag].dropna().unique()).issubset({0, 1}):
        selection = st.sidebar.selectbox(
            format_name(flag), ["All", "Yes only", "No only"], key=f"filter_{flag}"
        )
        if selection == "Yes only":
            filtered = filtered[filtered[flag] == 1]
        elif selection == "No only":
            filtered = filtered[filtered[flag] == 0]

st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} labelled records")

if filtered.empty:
    st.warning("No records match these filters. Reset one or more sidebar filters.")
    st.stop()

# =========================================================
# HEADER AND KPI SUMMARY
# =========================================================
st.markdown(
    f"""
<div class="hero">
  <h1>🩺 Diabetes Health Analytics</h1>
  <p>Explore diabetes labels across age, body mass index, blood glucose, HbA1c and lifestyle indicators.</p>
  <span class="chip">Python</span><span class="chip">Streamlit</span>
  <span class="chip">Plotly</span><span class="chip">Kaggle dataset</span>
</div>
""",
    unsafe_allow_html=True,
)

number_of_records = len(filtered)
positive_cases = int(filtered["_diabetic"].sum())
positive_rate = positive_cases / number_of_records * 100

kpi_columns = st.columns(6)
kpi_values = [
    ("Records", f"{number_of_records:,}", "current filtered view", BLUE),
    ("Positive cases", f"{positive_cases:,}", "rows labelled diabetic", CORAL),
    ("Positive rate", f"{positive_rate:.1f}%", "within current view", CORAL),
    ("Average age", safe_mean(filtered, age_col, 1), "years", TEAL),
    ("Average BMI", safe_mean(filtered, bmi_col, 1), "kg/m²", AMBER),
    ("Average HbA1c", safe_mean(filtered, hba1c_col, 2), "dataset units", GREEN),
]
for column, (label, value, detail, accent) in zip(kpi_columns, kpi_values):
    with column:
        kpi_card(label, value, detail, accent)

st.caption(
    "Positive rate is calculated from the target labels in the selected dataset. "
    "It is not an estimate of the diabetes rate in the general population."
)

# =========================================================
# AUTOMATIC INSIGHTS
# =================================================n
insights = []
for group_col, description in [
    ("_age_group", "age group"),
    ("_bmi_group", "BMI category"),
    ("_hba1c_group", "HbA1c range"),
    ("_glucose_group", "blood glucose range"),
]:
    if group_col in filtered.columns:
        table = prevalence_table(filtered, group_col)
        table = table[table["Records"] >= 20]
        if not table.empty:
            top = table.sort_values("Diabetes Rate (%)", ascending=False).iloc[0]
            insights.append(
                f"The highest positive-label rate by {description} is "
                f"<b>{top[group_col]}</b> ({top['Diabetes Rate (%)']:.1f}%, "
                f"{int(top['Cases'])} of {int(top['Records'])} records)."
            )
for flag in ["hypertension", "heart_disease"]:
    if flag in filtered.columns and set(filtered[flag].dropna().unique()).issubset({0, 1}):
        yes_rows = filtered[filtered[flag] == 1]
        no_rows = filtered[filtered[flag] == 0]
        if len(yes_rows) >= 10 and len(no_rows) >= 10:
            yes_rate = yes_rows["_diabetic"].mean() * 100
            no_rate = no_rows["_diabetic"].mean() * 100
            insights.append(
                f"Positive-label rate: {yes_rate:.1f}% with {format_name(flag).lower()} "
                f"and {no_rate:.1f}% without it. This is an association, not proof of cause."
            )

with st.expander("✨ Key findings in the current filtered data", expanded=True):
    if insights:
        for insight in insights[:6]:
            st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)
    else:
        st.info("No automatic findings could be calculated from the available mapped columns.")

# =========================================================
# TABS
# =========================================================
tab_overview, tab_risk, tab_clinical, tab_lifestyle, tab_corr, tab_compare, tab_data, tab_about = st.tabs(
    ["🏠 Overview", "🎯 Risk Factors", "🧪 Clinical Indicators", "🚬 Lifestyle & Conditions",
     "🔗 Correlations", "🧍 Profile Comparison", "🗂️ Data Explorer", "📚 About"]
)

# ------------------------- OVERVIEW -------------------------
with tab_overview:
    left, middle, right = st.columns([1, 1, 1.25])
    with left:
        counts = filtered["_status"].value_counts().rename_axis("Status").reset_index(name="Records")
        fig = px.pie(
            counts, names="Status", values="Records", hole=0.58,
            color="Status", color_discrete_map=STATUS_COLORS,
            title="Target-label distribution",
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False)
        show_chart(fig, 350)
    with middle:
        gauge_max = max(100, positive_rate * 1.15)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=positive_rate,
            number={"suffix": "%", "font": {"color": NAVY}},
            title={"text": "Positive-label rate", "font": {"size": 15, "color": NAVY}},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": CORAL},
                "steps": [
                    {"range": [0, gauge_max * 0.33], "color": "#DDF3E8"},
                    {"range": [gauge_max * 0.33, gauge_max * 0.66], "color": "#FCEBC8"},
                    {"range": [gauge_max * 0.66, gauge_max], "color": "#FADADD"},
                ],
            },
        ))
        show_chart(fig, 350)
    with right:
        if age_col:
            fig = px.histogram(
                filtered, x=age_col, color="_status", nbins=35, barmode="overlay", opacity=0.72,
                color_discrete_map=STATUS_COLORS, title="Age distribution by target label",
                labels={"_status": "Target label"},
            )
            fig.update_yaxes(title="Records")
            show_chart(fig, 350)
        else:
            st.info("Map an age column in the sidebar to show the age distribution.")

    if gender_col:
        c1, c2 = st.columns(2)
        with c1:
            prevalence_chart(filtered, gender_col, f"Positive-label rate by {format_name(gender_col)}", positive_rate)
        with c2:
            gender_counts = filtered.groupby([gender_col, "_status"], observed=True).size().reset_index(name="Records")
            fig = px.bar(
                gender_counts, x=gender_col, y="Records", color="_status", barmode="group",
                color_discrete_map=STATUS_COLORS, title="Record count by gender and target label",
                labels={"_status": "Target label"}, text_auto=True,
            )
            show_chart(fig, 350)

# ------------------------- RISK FACTORS -------------------------
with tab_risk:
    info_note("Bars show the percentage of records labelled diabetic within each group. "
              "The dashed line is the overall positive-label rate in the current filtered data.")
    c1, c2 = st.columns(2)
    with c1:
        if "_age_group" in filtered.columns:
            prevalence_chart(filtered, "_age_group", "Positive-label rate by age group", positive_rate)
        else:
            st.info("Map an age column to enable age-group analysis.")
    with c2:
        if "_bmi_group" in filtered.columns:
            prevalence_chart(filtered, "_bmi_group", "Positive-label rate by BMI category", positive_rate)
        else:
            st.info("Map a BMI column to enable BMI-category analysis.")

    if "_age_group" in filtered.columns and "_bmi_group" in filtered.columns:
        pivot = filtered.pivot_table(
            index="_bmi_group", columns="_age_group", values="_diabetic", aggfunc="mean", observed=True
        ) * 100
        pivot = pivot.reindex(index=[x for x in BMI_LABELS if x in pivot.index],
                              columns=[x for x in AGE_LABELS if x in pivot.columns])
        if not pivot.empty:
            fig = px.imshow(
                pivot, text_auto=".1f", aspect="auto", color_continuous_scale=RISK_SCALE,
                title="Positive-label rate (%) by age group and BMI category",
                labels={"color": "Rate (%)", "x": "Age group", "y": "BMI category"},
            )
            show_chart(fig, 420)
            info_note("Check the number of records in each group before interpreting extreme percentages. "
                      "A small group can produce an unstable rate.")

    numeric_features = [
        col for col in filtered.select_dtypes(include=np.number).columns
        if col not in {target_column, "_diabetic"} and filtered[col].nunique(dropna=True) > 1
    ]
    if numeric_features:
        associations = filtered[numeric_features].corrwith(filtered["_diabetic"]).dropna().sort_values()
        if not associations.empty:
            ranking = associations.rename("Correlation with positive label").reset_index()
            ranking.columns = ["Feature", "Correlation"]
            ranking["Feature"] = ranking["Feature"].map(format_name)
            fig = px.bar(
                ranking, x="Correlation", y="Feature", orientation="h", color="Correlation",
                color_continuous_scale=[[0, BLUE], [0.5, "#E9EEF2"], [1, CORAL]],
                range_color=[-1, 1], title="Numeric feature association with the target label",
            )
            fig.update_coloraxes(showscale=False)
            show_chart(fig, max(320, 28 * len(ranking) + 100))
            info_note("Correlation measures association, not causation. Some clinical measurements may be closely related to the target because they are used to define it.")

    feature_candidates = [
        col for col in filtered.columns
        if col not in {"_diabetic", "_status", target_column}
        and not col.startswith("_")
        and filtered[col].nunique(dropna=True) <= 20
    ]
    if feature_candidates:
        chosen_feature = st.selectbox("Explore positive-label rate by another feature", feature_candidates)
        prevalence_chart(filtered, chosen_feature, f"Positive-label rate by {format_name(chosen_feature)}", positive_rate)

# ------------------------- CLINICAL INDICATORS -------------------------
with tab_clinical:
    clinical_columns = [col for col in [bmi_col, glucose_col, hba1c_col, age_col] if col]
    if clinical_columns:
        selected_clinical = st.selectbox("Choose a health indicator", clinical_columns,
                                         format_func=format_name)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                filtered, x=selected_clinical, color="_status", nbins=35, barmode="overlay",
                opacity=0.7, marginal="box", color_discrete_map=STATUS_COLORS,
                labels={"_status": "Target label"}, title=f"{format_name(selected_clinical)} distribution",
            )
            show_chart(fig, 420)
        with c2:
            fig = px.violin(
                filtered, x="_status", y=selected_clinical, color="_status", box=True,
                color_discrete_map=STATUS_COLORS, labels={"_status": "Target label"},
                title=f"{format_name(selected_clinical)} by target label",
            )
            fig.update_layout(showlegend=False)
            show_chart(fig, 420)
    else:
        st.info("Map one or more numeric clinical columns in the sidebar.")

    c1, c2 = st.columns(2)
    with c1:
        if "_hba1c_group" in filtered.columns:
            prevalence_chart(filtered, "_hba1c_group", "Positive-label rate by HbA1c group", positive_rate)
    with c2:
        if "_glucose_group" in filtered.columns:
            prevalence_chart(filtered, "_glucose_group", "Positive-label rate by glucose group", positive_rate)

    if hba1c_col and glucose_col:
        sample = filtered.sample(min(5000, len(filtered)), random_state=11)
        fig = px.scatter(
            sample, x=hba1c_col, y=glucose_col, color="_status", opacity=0.58,
            color_discrete_map=STATUS_COLORS, labels={"_status": "Target label"},
            title="HbA1c compared with blood glucose (sample up to 5,000 records)",
        )
        show_chart(fig, 440)
    info_note("Clinical reference cut-offs are simplified educational references. Actual diagnosis depends on the test type, symptoms and clinical confirmation.")

# ------------------------- LIFESTYLE & CONDITIONS -------------------------
with tab_lifestyle:
    c1, c2 = st.columns(2)
    with c1:
        if smoking_col:
            prevalence_chart(filtered, smoking_col, "Positive-label rate by smoking history", positive_rate)
        else:
            st.info("No smoking column mapped.")
    with c2:
        condition_columns = [c for c in ["hypertension", "heart_disease"] if c in filtered.columns]
        if condition_columns:
            selected_condition = st.selectbox("Choose a condition", condition_columns, format_func=format_name)
            condition_data = filtered.copy()
            condition_data[selected_condition] = condition_data[selected_condition].map({0: "No", 1: "Yes"}).fillna(condition_data[selected_condition].astype(str))
            prevalence_chart(condition_data, selected_condition, f"Positive-label rate by {format_name(selected_condition)}", positive_rate)
        else:
            st.info("No hypertension or heart-disease columns found.")

    categorical_columns = [
        c for c in filtered.columns
        if c not in {"_diabetic", "_status", target_column, smoking_col, gender_col}
        and not c.startswith("_")
        and filtered[c].nunique(dropna=True) <= 12
        and not pd.api.types.is_numeric_dtype(filtered[c])
    ]
    if categorical_columns:
        chosen_category = st.selectbox("Explore another lifestyle or categorical feature", categorical_columns)
        prevalence_chart(filtered, chosen_category, f"Positive-label rate by {format_name(chosen_category)}", positive_rate)

# ------------------------- CORRELATIONS -------------------------
with tab_corr:
    numeric_for_corr = [
        col for col in filtered.select_dtypes(include=np.number).columns
        if col != target_column and filtered[col].nunique(dropna=True) > 1
    ]
    corr_data = filtered[numeric_for_corr].copy()
    corr_data["diabetes_target_flag"] = filtered["_diabetic"]
    corr_options = corr_data.columns.tolist()
    default_corr = corr_options[:min(10, len(corr_options))]
    selected_corr = st.multiselect("Choose numeric features", corr_options, default=default_corr)
    if len(selected_corr) >= 2:
        matrix = corr_data[selected_corr].corr()
        fig = px.imshow(
            matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto", title="Correlation heatmap",
        )
        show_chart(fig, 500)
        info_note("+1 means a strong positive linear relationship, -1 means a strong negative linear relationship, and 0 means little or no linear relationship. Correlation does not prove cause.")
    else:
        st.info("Choose at least two numeric features to build the heatmap.")

    if len(numeric_for_corr) >= 2:
        x_col, y_col = st.columns(2)
        with x_col:
            x_feature = st.selectbox("X-axis", numeric_for_corr, index=0, key="corr_x")
        with y_col:
            y_feature = st.selectbox("Y-axis", numeric_for_corr, index=1, key="corr_y")
        sample = filtered.sample(min(5000, len(filtered)), random_state=17)
        fig = px.scatter(
            sample, x=x_feature, y=y_feature, color="_status", opacity=0.55,
            color_discrete_map=STATUS_COLORS, labels={"_status": "Target label"},
            title=f"{format_name(x_feature)} vs {format_name(y_feature)}",
        )
        show_chart(fig, 440)

# ------------------------- PROFILE COMPARISON -------------------------
with tab_compare:
    st.markdown("### Compare example measurements with records in this dataset")
    info_note("This section compares entered values with similar records in the dataset. It is not a medical risk score, diagnosis or prediction.")
    m1, m2, m3, m4 = st.columns(4)
    user_age = m1.number_input("Age (years)", min_value=1, max_value=120, value=40) if age_col else None
    user_bmi = m2.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1) if bmi_col else None
    user_a1c = m3.number_input("HbA1c", min_value=2.0, max_value=20.0, value=5.5, step=0.1) if hba1c_col else None
    user_glucose = m4.number_input("Blood glucose", min_value=20, max_value=600, value=110) if glucose_col else None

    comparable = df.copy()
    if age_col:
        comparable = comparable[comparable[age_col].between(user_age - 5, user_age + 5)]
    if bmi_col:
        comparable = comparable[comparable[bmi_col].between(user_bmi - 3, user_bmi + 3)]
    if hba1c_col:
        comparable = comparable[comparable[hba1c_col].between(user_a1c - 0.5, user_a1c + 0.5)]
    if glucose_col:
        comparable = comparable[comparable[glucose_col].between(user_glucose - 30, user_glucose + 30)]

    if not any([age_col, bmi_col, hba1c_col, glucose_col]):
        st.info("Map at least one health indicator in the sidebar to use profile comparison.")
    elif len(comparable) >= 20:
        similar_rate = comparable["_diabetic"].mean() * 100
        c1, c2 = st.columns([1, 1.4])
        with c1:
            kpi_card("Similar records", f"{len(comparable):,}", "matching the selected ranges", BLUE)
            st.write("")
            kpi_card("Positive-label rate", f"{similar_rate:.1f}%", "among similar records", AMBER)
        with c2:
            percentile_rows = []
            for label, column, value in [
                ("Age", age_col, user_age), ("BMI", bmi_col, user_bmi),
                ("HbA1c", hba1c_col, user_a1c), ("Glucose", glucose_col, user_glucose),
            ]:
                if column:
                    values = pd.to_numeric(df[column], errors="coerce").dropna()
                    if not values.empty:
                        percentile_rows.append({"Measure": label, "Percentile": (values < value).mean() * 100})
            if percentile_rows:
                percentiles = pd.DataFrame(percentile_rows)
                fig = px.bar(
                    percentiles, x="Percentile", y="Measure", orientation="h", range_x=[0, 100],
                    text=percentiles["Percentile"].map(lambda v: f"{v:.0f}th"),
                    color="Percentile", color_continuous_scale=RISK_SCALE,
                    title="Where the entered values sit in this dataset",
                )
                fig.update_coloraxes(showscale=False)
                show_chart(fig, 300)
    else:
        st.warning("Fewer than 20 similar records were found. Widen the comparison ranges or use different example values.")

# ------------------------- DATA EXPLORER -------------------------
with tab_data:
    st.markdown("### Explore and download the dataset")
    search_text = st.text_input("🔍 Search records", placeholder="Search a value or category")
    display_data = filtered.drop(columns=["_diabetic"]).rename(columns={"_status": "dashboard_label"}).copy()
    if search_text:
        matches = display_data.astype(str).apply(
            lambda series: series.str.contains(search_text, case=False, na=False, regex=False)
        ).any(axis=1)
        display_data = display_data[matches]
    st.write(f"Showing **{len(display_data):,}** records")
    st.dataframe(display_data, use_container_width=True, height=380)
    st.download_button(
        "⬇ Download filtered records",
        data=display_data.to_csv(index=False).encode("utf-8"),
        file_name="diabetes_dashboard_filtered.csv",
        mime="text/csv",
    )
    with st.expander("📋 Statistical summary"):
        st.dataframe(filtered.describe(include="all").transpose(), use_container_width=True)
    missing = filtered.isna().sum()
    missing = missing[missing > 0].rename_axis("Feature").reset_index(name="Missing values")
    if missing.empty:
        st.success("No missing values in the currently filtered records.")
    else:
        fig = px.bar(missing, x="Feature", y="Missing values", color="Missing values",
                     color_continuous_scale="Reds", title="Missing values by feature")
        show_chart(fig, 330)

# ------------------------- ABOUT / INTERPRETATION -------------------------
with tab_about:
    st.markdown("### Project overview")
    st.write(
        "This dashboard explores patterns in a Kaggle diabetes dataset. "
        "It uses descriptive statistics and interactive visualizations to help viewers inspect the data."
    )
    st.markdown("### Main project stages")
    st.markdown(
        "1. **Data acquisition:** download the dataset from Kaggle.\n"
        "2. **Data preparation:** standardize column names, convert numeric fields and handle missing target labels.\n"
        "3. **Exploratory analysis:** calculate counts, percentages, averages and correlations.\n"
        "4. **Visualization:** present results using Plotly charts.\n"
        "5. **Interactive exploration:** filter records, compare groups and export the filtered data."
    )
    st.markdown("### How to interpret the dashboard")
    st.markdown(
        "- **Overview:** summarizes the selected records and target-label split.\n"
        "- **Risk Factors:** compares positive-label percentages across groups.\n"
        "- **Clinical Indicators:** compares numerical measurements between target groups.\n"
        "- **Lifestyle & Conditions:** explores smoking and available health-condition fields.\n"
        "- **Correlations:** shows linear relationships between numeric variables.\n"
        "- **Profile Comparison:** compares example values with similar records in the dataset."
    )
    st.markdown("### Dataset details")
    st.write(f"**Source:** Kaggle — `{DATASET_NAME}`")
    st.write(f"**Selected file:** `{Path(selected_file).name}`")
    st.write(f"**Rows used:** {len(df):,} labelled records")
    st.write(f"**Columns:** {len(raw_df.columns)}")
    st.write("**Target column:**", target_column)
    st.write("**Positive target value(s):**", ", ".join(map(str, positive_values)))

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(
    f"""
<div style="text-align:center;color:#657789;line-height:1.8;">
  <b style="color:{NAVY};font-size:1.05rem;">Diabetes Health Analytics</b><br>
  Python · Streamlit · Plotly · KaggleHub<br>
  Academic Data Science Project
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    "Educational data-analysis project only. This dashboard does not diagnose diabetes, "
    "estimate an individual's medical risk, or replace advice from a qualified healthcare professional."
)
