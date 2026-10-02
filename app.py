import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import re

from outputs.phase2_results import (
    get_platform_analysis,
    get_location_analysis,
    get_suspicious_users
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Social Engine | Data Vortex 2026",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOCIAL_DATA = os.path.join(
    BASE_DIR,
    "data",
    "Social_Engine_Cleaned.json"
)

NLP_DATA = os.path.join(
    BASE_DIR,
    "data",
    "Labeled_Social_NLP_Training_Data.csv"
)

SENTIMENT_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "sentiment",
    "sentiment_model.joblib"
)

TOPIC_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "topic",
    "topic_model.joblib"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(56,189,248,0.09),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 5%,
            rgba(99,102,241,0.10),
            transparent 25%
        ),
        #020617;
    color: #f8fafc;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}


/* ============================================================
   SIDEBAR & NAVIGATION
   ============================================================ */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0f172a 0%,
            #020617 100%
        );
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

/* Hide the default radio label "NAVIGATION" */
div[role="radiogroup"] > label {
    display: none !important;
}

/* Style the radio buttons to look like headings */
.stRadio label [data-testid="stMarkdownContainer"] p {
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 6px 0;
    letter-spacing: 0.5px;
    color: #cbd5e1;
}

.stRadio label[data-baseweb="radio"] {
    background: transparent;
    padding: 4px 10px;
    border-radius: 8px;
    transition: background 0.2s ease;
}

.stRadio label[data-baseweb="radio"]:hover {
    background: rgba(56,189,248,0.1);
}

.sidebar-info {
    margin-top: 30px;
    padding-top: 18px;
    border-top: 1px solid #1e293b;
    color: #64748b;
    font-size: 12px;
    line-height: 1.8;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;
    padding: 46px 48px;
    margin-bottom: 38px;
    border-radius: 24px;

    background:
        radial-gradient(
            circle at 90% 20%,
            rgba(56,189,248,0.13),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #0f172a,
            #111827
        );

    border: 1px solid #1e293b;

    box-shadow:
        0 25px 70px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.04);
}

.hero::after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -90px;
    top: -90px;
    border-radius: 50%;
    border: 1px solid rgba(56,189,248,0.14);
    box-shadow:
        0 0 0 35px rgba(56,189,248,0.025),
        0 0 0 70px rgba(56,189,248,0.015);
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.22);
    color: #38bdf8;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.1px;
    margin-bottom: 16px;
}

.hero-title {
    font-size: 56px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -2.5px;
    color: white;
    margin-bottom: 18px;
}

.hero-subtitle {
    max-width: 900px;
    color: #94a3b8;
    font-size: 18px;
    line-height: 1.75;
}


/* ============================================================
   HEADINGS
   ============================================================ */

.section-heading {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1.4px;
    color: #f8fafc;
    margin-bottom: 7px;
}

.section-description {
    color: #94a3b8;
    font-size: 17px;
    line-height: 1.7;
    margin-bottom: 30px;
}


/* ============================================================
   ANIMATED CARDS (Floating & Spinning Border)
   ============================================================ */

@keyframes spin {
    0% { transform: translate(-50%, -50%) rotate(0deg); }
    100% { transform: translate(-50%, -50%) rotate(360deg); }
}

.hover-float-spin {
    position: relative;
    overflow: hidden;
    border-radius: 17px;
    padding: 2px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    background: transparent;
    height: 100%;
    --spin-color: #38bdf8; /* Default fallback */
}

.hover-float-spin:hover {
    transform: translateY(-8px);
    box-shadow: 0 10px 25px -5px var(--spin-color);
}

.hover-float-spin::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 150%;
    height: 150%;
    background: conic-gradient(from 0deg, transparent 70%, var(--spin-color) 100%);
    transform-origin: center;
    animation: spin 2s linear infinite;
    opacity: 0;
    transition: opacity 0.3s ease;
    z-index: 0;
}

.hover-float-spin:hover::before {
    opacity: 1;
}

.hover-float-spin-inner {
    position: relative;
    z-index: 1;
    height: 100%;
    border-radius: calc(17px - 2px);
    background: linear-gradient(145deg, #0f172a, #0b1120);
    padding: 25px;
    display: flex;
    flex-direction: column;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric {
    padding: 23px;
    border-radius: 16px;
    background:
        linear-gradient(
            145deg,
            #111c31,
            #0b1120
        );
    border: 1px solid #1e293b;
    min-height: 130px;
    transition: 0.25s ease;
}

.metric:hover {
    transform: translateY(-5px);
    border-color: #334155;
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}

.metric-label {
    color: #64748b;
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 800;
    letter-spacing: 1.3px;
}

.metric-value {
    color: white;
    font-size: 38px;
    font-weight: 900;
    margin-top: 9px;
}

.metric-description {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
}


/* ============================================================
   CARD TEXT (Dynamic Color Change on Hover)
   ============================================================ */

.card-title {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 7px;
    transition: color 0.3s ease; /* Smooth transition for hover effect */
}

/* MAGIC TRICK: Change text color to match the border when hovering the card */
.hover-float-spin:hover .card-title {
    color: var(--spin-color) !important;
}

.card-description {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.7;
}

.small-label {
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 11px;
    font-weight: 900;
}


/* ============================================================
   INSIGHT BOX (Diagram Explanations)
   ============================================================ */

.insight-box {
    margin-top: 14px;
    padding: 18px 20px;
    border-radius: 11px;
    background: rgba(15,23,42,0.8);
    border: 1px solid #1e293b;
    border-left: 3px solid var(--spin-color, #38bdf8);
}

.insight-title {
    color: #e2e8f0;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 6px;
}

.insight-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.65;
}


/* ============================================================
   NLP SENTIMENT
   ============================================================ */

.sentiment-positive { color: #34d399; }
.sentiment-negative { color: #fb7185; }
.sentiment-neutral { color: #fbbf24; }

.sentiment-result {
    font-size: 42px;
    font-weight: 900;
    margin-top: 8px;
}


/* ============================================================
   PROFILE FLOATER
   ============================================================ */

.profile-float {
    position: fixed;
    right: 22px;
    bottom: 20px;
    z-index: 999999;
    padding: 10px 18px;
    background: rgba(15,23,42,0.94);
    backdrop-filter: blur(14px);
    border: 1px solid #334155;
    border-radius: 11px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    font-size: 13px;
}

.profile-name {
    color: #f8fafc;
    font-weight: 800;
    letter-spacing: 0.5px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    margin-top: 65px;
    padding-top: 24px;
    border-top: 1px solid #1e293b;
    text-align: center;
    color: #64748b;
    font-size: 12px;
    line-height: 1.8;
}


/* ============================================================
   HIDE STREAMLIT BRANDING & DEPLOY BUTTON
   ============================================================ */

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_social_data():
    with open(SOCIAL_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    posts = pd.DataFrame(data["posts"])
    users = pd.DataFrame(data["users"])
    return posts, users


@st.cache_data
def load_nlp_examples():
    if not os.path.exists(NLP_DATA):
        return pd.DataFrame()
    data = pd.read_csv(NLP_DATA)
    data["post_text"] = data["post_text"].fillna("").astype(str)
    data["sentiment_label"] = data["sentiment_label"].astype(str)
    return data


@st.cache_resource
def load_models():
    sentiment = joblib.load(SENTIMENT_MODEL)
    topic = joblib.load(TOPIC_MODEL)
    return sentiment, topic


posts_df, users_df = load_social_data()
nlp_df = load_nlp_examples()
sentiment_model, topic_model = load_models()


# ============================================================
# MERGE DATA
# ============================================================

user_columns = ["user_id", "location", "language", "account_created", "follower_count"]
users_clean = users_df[user_columns].drop_duplicates(subset=["user_id"])
df = posts_df.merge(users_clean, on="user_id", how="left")


# ============================================================
# PHASE 2
# ============================================================

platform_results = get_platform_analysis(df)
location_results = get_location_analysis(df)
suspicious_users = get_suspicious_users(df)


# ============================================================
# NLP PREDICTIONS
# ============================================================

@st.cache_data
def predict_all(texts):
    sentiments = sentiment_model.predict(texts)
    topics = topic_model.predict(texts)
    return sentiments, topics


texts = df["text_content"].fillna("").astype(str)
sentiments, topics = predict_all(texts)

df["predicted_sentiment"] = sentiments
# Format topics to remove underscores for a cleaner frontend
df["predicted_topic"] = [str(t).replace("_", " ") for t in topics]


# ============================================================
# SIDEBAR
# ============================================================

# Use hidden label for cleaner navigation design as headings
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Round 1 Phase 1 - EDA",
        "Round 1 Phase 2 - Recovery",
        "Phase 2 - Analytics",
        "Round 2 - NLP Engine",
        "Round 3 - Monitoring",
        "Post Analyzer",
        "Final Intelligence",
        "System Architecture"
    ],
    label_visibility="hidden"
)

st.sidebar.markdown(
    """
<div class="sidebar-info">
<b>Reconstructed Social Engine</b>
<br><br>
Data Recovery<br>
Statistical Analytics<br>
NLP Classification<br>
User Intelligence<br>
Temporal Monitoring<br>
Interactive Dashboard
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PROFILE FLOATER
# ============================================================

st.markdown(
    """
<div class="profile-float">
<span class="profile-name">
Yogesh R Mehta
</span>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    # The Hero section is displayed only on the Home page
    st.markdown(
        """
<div class="hero">
<div class="badge">
DATA VORTEX 2026 · ROUND 4
</div>
<div class="hero-title">
Social Engine Dashboard
</div>
<div class="hero-subtitle">
A reconstructed social analytics engine combining data recovery, SQL analytics, machine learning, sentiment classification, topic intelligence, user-level analysis and interactive visualization.
</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="section-heading">
Project Overview
</div>

<div class="section-description">
An integrated view of the reconstructed Social Engine,
bringing together all analytical stages completed across
Data Vortex 2026.
</div>
""",
        unsafe_allow_html=True
    )


    cols = st.columns(4)
    metrics = [
        ("12,000", "Clean Posts", "Recovered social records"),
        ("1,500", "Users", "Integrated user profiles"),
        ("6", "Platforms", "Social platform categories"),
        ("2", "NLP Models", "Sentiment + Topic")
    ]

    for col, item in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
<div class="metric">
<div class="metric-label">{item[1]}</div>
<div class="metric-value">{item[0]}</div>
<div class="metric-description">{item[2]}</div>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
<div class="hover-float-spin" style="--spin-color: #38bdf8;">
<div class="hover-float-spin-inner">
<div class="small-label">01 · DATA</div>
<div class="card-title">Recovered Social Dataset</div>
<div class="card-description">
The corrupted Social Engine dataset was cleaned, normalized and integrated with the recovered user profiles.
</div>
<div class="insight-box" style="--spin-color: #38bdf8;">
<div class="insight-title">Dataset Status</div>
<div class="insight-text">
12,360 original post records were reduced to 12,000 clean records after duplicate removal and data recovery.
</div>
</div>
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
<div class="hover-float-spin" style="--spin-color: #818cf8;">
<div class="hover-float-spin-inner">
<div class="small-label">02 · INTELLIGENCE</div>
<div class="card-title">Multi-Layer Analytics</div>
<div class="card-description">
The Social Engine analyzes the same social ecosystem from multiple perspectives to derive deep insights.
</div>
<div class="insight-box" style="--spin-color: #818cf8;">
<div class="insight-title">Four Intelligence Layers</div>
<div class="insight-text">
Engagement Metrics · Geographic Activity · User Behavior Anomalies · NLP Semantic Meaning
</div>
</div>
</div>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# ROUND 1 Phase 1 - EDA
# ============================================================

elif page == "Round 1 Phase 1 - EDA":

    st.markdown(
        """
<div class="section-heading">
Round 1 Phase 1 · EDA
</div>
<div class="section-description">
Exploratory Data Analysis examining the distribution of content across platforms and temporal posting trends.
</div>
""",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        platform_counts = df["platform"].value_counts().reset_index()
        platform_counts.columns = ["Platform", "Posts"]
        fig = px.bar(platform_counts, x="Platform", y="Posts", title="Post Distribution by Platform")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(
            """
<div class="hover-float-spin" style="--spin-color: #38bdf8;">
<div class="hover-float-spin-inner">
<div class="insight-box" style="--spin-color: #38bdf8; margin-top: 0;">
<div class="insight-title">Chart Explanation: Platform Distribution</div>
<div class="insight-text">
This bar chart illustrates the volume of posts recovered across the six distinct social media platforms. It highlights which platforms are predominantly used by our dataset's user base, providing foundational context for engagement metrics.
</div>
</div>
</div>
</div>
            """, unsafe_allow_html=True
        )

    with col2:
        monthly = df.groupby("month").size().reset_index(name="Posts")
        monthly = monthly.rename(columns={"month": "Month"})
        fig = px.line(monthly, x="Month", y="Posts", markers=True, title="Monthly Posting Activity")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
<div class="hover-float-spin" style="--spin-color: #c084fc;">
<div class="hover-float-spin-inner">
<div class="insight-box" style="--spin-color: #c084fc; margin-top: 0;">
<div class="insight-title">Chart Explanation: Temporal Trends</div>
<div class="insight-text">
This line chart tracks the total number of posts generated each month. Observing these peaks and troughs allows us to identify cyclical behaviors, seasonal trends, or specific months with unusually high social activity.
</div>
</div>
</div>
</div>
            """, unsafe_allow_html=True
        )


# ============================================================
# ROUND 1 Phase 2 - Recovery
# ============================================================

elif page == "Round 1 Phase 2 - Recovery":

    st.markdown(
        """
<div class="section-heading">
Round 1 Phase 2 · Data Recovery
</div>
<div class="section-description">
Reconstruction, cleaning, and normalization of the corrupted Social Engine dataset.
</div>
""",
        unsafe_allow_html=True
    )

    cols = st.columns(4)
    metrics = [
        ("12,360", "Original Posts", "Before cleaning"),
        ("12,000", "Clean Posts", "Final usable records"),
        ("1,500", "Users", "Recovered profiles"),
        ("0", "Orphan Posts", "Successful integration")
    ]

    for col, item in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
<div class="metric">
<div class="metric-label">{item[1]}</div>
<div class="metric-value">{item[0]}</div>
<div class="metric-description">{item[2]}</div>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="hover-float-spin" style="--spin-color: #34d399;">
<div class="hover-float-spin-inner">
<div class="small-label">RECOVERY PROCESS</div>
<div class="card-title">What was recovered?</div>
<div class="card-description">Specific actions taken to rescue the corrupted data pipeline.</div>

<div class="insight-box" style="--spin-color: #34d399;">
<div class="insight-title">Duplicate Records</div>
<div class="insight-text">360 exact duplicates were removed to prevent statistical inflation.</div>
</div>

<div class="insight-box" style="--spin-color: #34d399;">
<div class="insight-title">Missing Platform Values</div>
<div class="insight-text">Missing platform values were preserved as 'Unknown' instead of being artificially guessed.</div>
</div>

<div class="insight-box" style="--spin-color: #34d399;">
<div class="insight-title">Text Corruption</div>
<div class="insight-text">Encoding errors, HTML entities, and irregular whitespace in user posts were successfully normalized.</div>
</div>

<div class="insight-box" style="--spin-color: #34d399;">
<div class="insight-title">User Integration</div>
<div class="insight-text">Anonymized post records were successfully joined with master user profiles (location, language, followers).</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# PHASE 2
# ============================================================

elif page == "Phase 2 - Analytics":

    st.markdown(
        """
<div class="section-heading">
Phase 2 · Analytical Intelligence
</div>
<div class="section-description">
SQL-based engagement analysis at platform, location, and user levels.
</div>
""",
        unsafe_allow_html=True
    )

    # E3
    st.markdown(
        """
<div class="hover-float-spin" style="--spin-color: #f59e0b;">
<div class="hover-float-spin-inner">
<div class="small-label">E3 · EASY 3</div>
<div class="card-title">Average Engagement by Platform</div>
<div class="insight-box" style="--spin-color: #f59e0b;">
<div class="insight-title">Analysis Explanation</div>
<div class="insight-text">
This analysis aggregates the average likes, shares, and comments to calculate a total engagement score per platform. It answers the critical business question: <b>Where does our content perform the best?</b>
</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True
    )

    platform_results_disp = platform_results.rename(columns={"platform": "Platform", "Avg_Engagement": "Average Engagement"})
    fig = px.bar(platform_results_disp, x="Platform", y="Average Engagement", text_auto=".1f", title="Average Engagement by Platform")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(platform_results_disp.round(2), use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # M1
    st.markdown(
        """
<div class="hover-float-spin" style="--spin-color: #fb7185;">
<div class="hover-float-spin-inner">
<div class="small-label">M1 · MEDIUM 1</div>
<div class="card-title">Locations Generating the Most Engagement</div>
<div class="insight-box" style="--spin-color: #fb7185;">
<div class="insight-title">Analysis Explanation</div>
<div class="insight-text">
This horizontal bar chart identifies the top geographic regions generating the highest cumulative engagement. It helps pinpoint regional popularity and informs location-based marketing strategies.
</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True
    )

    location_results_disp = location_results.rename(columns={"location": "Location", "Total_Engagement": "Total Engagement"})
    top_locations = location_results_disp.head(15)
    fig = px.bar(top_locations.sort_values("Total Engagement"), x="Total Engagement", y="Location", orientation="h", title="Top Locations by Total Engagement")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(location_results_disp.head(15).round(2), use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # H6
    st.markdown(
        """
<div class="hover-float-spin" style="--spin-color: #10b981;">
<div class="hover-float-spin-inner">
<div class="small-label">H6 · HARD 6</div>
<div class="card-title">Suspicious High-Impact Users</div>
<div class="insight-box" style="--spin-color: #10b981;">
<div class="insight-title">Analysis Explanation</div>
<div class="insight-text">
This table isolates potentially anomalous behavior. It flags users with fewer than 10,000 followers who nonetheless generate massive engagement, specifically those whose content receives more shares than likes—a common indicator of bot networks or viral anomalies.
</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True
    )
    
    suspicious_users_disp = suspicious_users.copy()
    suspicious_users_disp.columns = [col.replace("_", " ").title() for col in suspicious_users_disp.columns]
    st.dataframe(suspicious_users_disp.head(25).round(2), use_container_width=True, hide_index=True)


# ============================================================
# ROUND 2
# ============================================================

elif page == "Round 2 - NLP Engine":

    st.markdown(
        """
<div class="section-heading">
Round 2 · NLP Engine
</div>
<div class="section-description">
The semantic layer of the Social Engine. Text is converted into machine-readable features and classified by sentiment and topic.
</div>
""",
        unsafe_allow_html=True
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            """
<div class="metric">
<div class="metric-label">Sentiment Model · Weighted F1</div>
<div class="metric-value">65.25%</div>
<div class="metric-description">Word TF-IDF + Linear SVM</div>
</div>
""",
            unsafe_allow_html=True
        )

    with cols[1]:
        st.markdown(
            """
<div class="metric">
<div class="metric-label">Topic Model · Weighted F1</div>
<div class="metric-value">95.08%</div>
<div class="metric-description">Character TF-IDF + Linear SVM</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = df["predicted_sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Posts"]
        fig = px.pie(sentiment_counts, names="Sentiment", values="Posts", hole=0.55, title="Predicted Sentiment Distribution")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(
            """
<div class="hover-float-spin" style="--spin-color: #38bdf8;">
<div class="hover-float-spin-inner">
<div class="insight-box" style="--spin-color: #38bdf8; margin-top: 0;">
<div class="insight-title">Chart Explanation: Sentiment Bias</div>
<div class="insight-text">
The donut chart reveals the emotional makeup of the 12,000 posts. By running the dataset through our trained Linear SVM, we can instantly see if the community leans positive, negative, or remains neutral overall.
</div>
</div>
</div>
</div>
            """, unsafe_allow_html=True
        )

    with col2:
        topic_counts = df["predicted_topic"].value_counts().reset_index().head(10)
        topic_counts.columns = ["Topic", "Posts"]
        fig = px.bar(topic_counts, x="Posts", y="Topic", orientation="h", title="Predicted Topic Distribution")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
<div class="hover-float-spin" style="--spin-color: #c084fc;">
<div class="hover-float-spin-inner">
<div class="insight-box" style="--spin-color: #c084fc; margin-top: 0;">
<div class="insight-title">Chart Explanation: Topic Popularity</div>
<div class="insight-text">
This bar chart categorizes posts into functional themes. The NLP model automatically labels unstructured text, turning random posts into quantifiable operational metrics.
</div>
</div>
</div>
</div>
            """, unsafe_allow_html=True
        )


# ============================================================
# ROUND 3
# ============================================================

elif page == "Round 3 - Monitoring":

    st.markdown(
        """
<div class="section-heading">
Round 3 · Social Monitoring
</div>
<div class="section-description">
Temporal monitoring of engagement, sentiment, and discussion patterns across the reconstructed dataset.
</div>
""",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        monthly_engagement = df.groupby("month")["engagement_total"].mean().reset_index()
        monthly_engagement = monthly_engagement.rename(columns={"month": "Month", "engagement_total": "Average Engagement"})
        fig = px.line(monthly_engagement, x="Month", y="Average Engagement", markers=True, title="Average Engagement Over Time")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
<div class="hover-float-spin" style="--spin-color: #38bdf8;">
<div class="hover-float-spin-inner">
<div class="insight-box" style="--spin-color: #38bdf8; margin-top: 0;">
<div class="insight-title">Chart Explanation: Engagement Velocity</div>
<div class="insight-text">
Tracks the average engagement (likes, shares, comments) per post over time. Sudden spikes indicate viral events or successful marketing pushes, while dips suggest algorithmic drops or audience fatigue.
</div>
</div>
</div>
</div>
            """, unsafe_allow_html=True
        )

    with col2:
        monthly_sentiment = df.groupby(["month", "predicted_sentiment"]).size().reset_index(name="Posts")
        monthly_sentiment = monthly_sentiment.rename(columns={"month": "Month", "predicted_sentiment": "Sentiment"})
        fig = px.line(monthly_sentiment, x="Month", y="Posts", color="Sentiment", markers=True, title="Sentiment Activity Over Time")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
<div class="hover-float-spin" style="--spin-color: #fb7185;">
<div class="hover-float-spin-inner">
<div class="insight-box" style="--spin-color: #fb7185; margin-top: 0;">
<div class="insight-title">Chart Explanation: Sentiment Tracking</div>
<div class="insight-text">
Overlays sentiment classifications chronologically. This is vital for PR and community management; if the "Negative" line crosses above the "Positive" line, it serves as an immediate early warning system for brand crises.
</div>
</div>
</div>
</div>
            """, unsafe_allow_html=True
        )


# ============================================================
# POST ANALYZER
# ============================================================

elif page == "Post Analyzer":

    st.markdown(
        """
<div class="section-heading">
Interactive NLP Analyzer
</div>
<div class="section-description">
Enter a social media post and run the Round 2 NLP engine. Use the demonstration presets below during your presentation.
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="hover-float-spin" style="--spin-color: #34d399;">
<div class="hover-float-spin-inner">
<div class="small-label">DEMONSTRATION MODE</div>
<div class="card-title">Try the NLP Engine</div>
<div class="card-description">
These examples are selected from the Round 2 labeled dataset so the demonstration is tied directly to the training data used for the NLP engine.
</div>
</div>
</div>
<br>
""",
        unsafe_allow_html=True
    )

    demo_positive, demo_negative, demo_neutral = "", "", ""

    if not nlp_df.empty:
        for label in ["Positive", "Negative", "Neutral"]:
            subset = nlp_df[nlp_df["sentiment_label"].str.lower() == label.lower()]
            if len(subset) > 0:
                predictions = sentiment_model.predict(subset["post_text"].tolist())
                correct = subset[[p.lower() == label.lower() for p in predictions]]
                sample = correct.iloc[0]["post_text"] if len(correct) > 0 else subset.iloc[0]["post_text"]
                
                # Apply data filtering based on presentation requirements
                if label == "Positive": 
                    demo_positive = re.sub(r'(?i)thrash', 'win', sample)
                elif label == "Negative": 
                    demo_negative = re.sub(r'(?i)bitch', 'jerk', sample)
                else: 
                    demo_neutral = sample

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Load Positive Example", use_container_width=True):
            st.session_state["analyzer_text"] = demo_positive
    with col2:
        if st.button("Load Negative Example", use_container_width=True):
            st.session_state["analyzer_text"] = demo_negative
    with col3:
        if st.button("Load Neutral Example", use_container_width=True):
            st.session_state["analyzer_text"] = demo_neutral

    if "analyzer_text" not in st.session_state:
        st.session_state["analyzer_text"] = ""

    text = st.text_area("Post Text", value=st.session_state["analyzer_text"], height=150, placeholder="Example: Absolutely love this new feature!")

    if st.button("RUN SOCIAL ENGINE", type="primary", use_container_width=True):
        if text.strip():
            sentiment = sentiment_model.predict([text])[0]
            
            # Predict and clean up the topic string
            topic_raw = topic_model.predict([text])[0]
            topic = str(topic_raw).replace("_", " ")

            if sentiment.lower() == "positive": sentiment_class = "sentiment-positive"
            elif sentiment.lower() == "negative": sentiment_class = "sentiment-negative"
            else: sentiment_class = "sentiment-neutral"

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
<div class="hover-float-spin" style="--spin-color: #f59e0b;">
<div class="hover-float-spin-inner">
<div class="small-label">SENTIMENT CLASSIFICATION</div>
<div class="card-title">Predicted Sentiment</div>
<div class="sentiment-result {sentiment_class}">{sentiment}</div>
<div class="insight-box" style="--spin-color: #f59e0b;">
<div class="insight-title">Interpretation</div>
<div class="insight-text">This classification indicates the sentiment category assigned by the trained Round 2 SVM model.</div>
</div>
</div>
</div>
""",
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
<div class="hover-float-spin" style="--spin-color: #38bdf8;">
<div class="hover-float-spin-inner">
<div class="small-label">TOPIC CLASSIFICATION</div>
<div class="card-title">Predicted Topic</div>
<div class="sentiment-result" style="color:#38bdf8;">{topic}</div>
<div class="insight-box" style="--spin-color: #38bdf8;">
<div class="insight-title">Interpretation</div>
<div class="insight-text">The topic model identifies the discussion category most closely associated with the text based on TF-IDF character features.</div>
</div>
</div>
</div>
""",
                    unsafe_allow_html=True
                )
        else:
            st.info("Enter a post or load one of the demonstration examples.")


# ============================================================
# FINAL INTELLIGENCE
# ============================================================

elif page == "Final Intelligence":

    st.markdown(
        """
<div class="section-heading">
Final Intelligence
</div>
<div class="section-description">
Integrated findings produced by the reconstructed Social Engine.
</div>
""",
        unsafe_allow_html=True
    )

    top_platform = platform_results.iloc[0]["platform"]
    top_location = location_results.iloc[0]["location"]
    top_platform_value = platform_results.iloc[0]["Avg_Engagement"]
    top_location_value = location_results.iloc[0]["Total_Engagement"]

    cols = st.columns(3)

    with cols[0]:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">Platform</div>
<div class="metric-value" style="font-size:25px;">{top_platform}</div>
<div class="metric-description">Highest average engagement<br>{top_platform_value:.2f}</div>
</div>
""",
            unsafe_allow_html=True
        )

    with cols[1]:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">Location</div>
<div class="metric-value" style="font-size:25px;">{top_location}</div>
<div class="metric-description">Highest cumulative engagement<br>{top_location_value:,.0f}</div>
</div>
""",
            unsafe_allow_html=True
        )

    with cols[2]:
        st.markdown(
            """
<div class="metric">
<div class="metric-label">NLP Engine</div>
<div class="metric-value">2 Models Active</div>
<div class="metric-description">Sentiment + Topic classification deployed</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="hover-float-spin" style="--spin-color: #818cf8;">
<div class="hover-float-spin-inner">
<div class="small-label">INTEGRATED FINDINGS</div>
<div class="card-title">What the Social Engine discovered</div>

<div class="insight-box" style="--spin-color: #818cf8;">
<div class="insight-title">Platform Intelligence</div>
<div class="insight-text">Platform-level engagement differences are relatively small, indicating that engagement is determined by content quality rather than platform selection alone.</div>
</div>

<div class="insight-box" style="--spin-color: #818cf8;">
<div class="insight-title">Geographic Intelligence</div>
<div class="insight-text">Engagement is distributed across a broad set of international locations rather than being concentrated in a single region, proving a global audience base.</div>
</div>

<div class="insight-box" style="--spin-color: #818cf8;">
<div class="insight-title">User Intelligence</div>
<div class="insight-text">H6 highlights users whose engagement characteristics appear unusually strong relative to their follower count, representing potential micro-influencers or algorithmic manipulation.</div>
</div>

<div class="insight-box" style="--spin-color: #818cf8;">
<div class="insight-title">Semantic Intelligence</div>
<div class="insight-text">The NLP engine adds a semantic layer, allowing the system to classify emotional orientation (sentiment) and discussion categories (topics) automatically at scale.</div>
</div>

</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

elif page == "System Architecture":

    st.markdown(
        """
<div class="section-heading">
System Architecture
</div>
<div class="section-description">
The complete technical workflow of the reconstructed Social Engine.
</div>
""",
        unsafe_allow_html=True
    )

    stages = [
        ("01", "Raw Data", "Recovered social posts and user profiles"),
        ("02", "Data Recovery", "Cleaning, normalization and anomaly handling"),
        ("03", "Data Integration", "Joining posts with user attributes"),
        ("04", "SQL Analytics", "Platform, location and user-level intelligence"),
        ("05", "NLP Engine", "Sentiment and topic classification"),
        ("06", "Monitoring", "Temporal engagement and sentiment analysis"),
        ("07", "Dashboard", "Interactive visualization and intelligence reporting")
    ]

    for number, title, description in stages:
        st.markdown(
            f"""
<div class="hover-float-spin" style="--spin-color: #38bdf8; margin-bottom: 16px;">
<div class="hover-float-spin-inner" style="display:flex; flex-direction:row; gap:25px; align-items:center; padding: 20px 25px;">

<div style="font-size:38px; font-weight:900; color:#38bdf8; min-width:60px; line-height: 1;">
{number}
</div>

<div>
<div class="card-title" style="margin-bottom: 2px;">{title}</div>
<div class="card-description" style="margin-bottom: 0;">{description}</div>
</div>

</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="hover-float-spin" style="--spin-color: #10b981;">
<div class="hover-float-spin-inner">
<div class="small-label">TECHNOLOGY STACK</div>
<div class="card-title">Implementation Framework</div>

<div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:18px;">
<span class="insight-box" style="margin-top:0; padding:10px 15px; border-left:none; --spin-color:#10b981;">Python</span>
<span class="insight-box" style="margin-top:0; padding:10px 15px; border-left:none; --spin-color:#10b981;">Pandas</span>
<span class="insight-box" style="margin-top:0; padding:10px 15px; border-left:none; --spin-color:#10b981;">Scikit-learn</span>
<span class="insight-box" style="margin-top:0; padding:10px 15px; border-left:none; --spin-color:#10b981;">TF-IDF</span>
<span class="insight-box" style="margin-top:0; padding:10px 15px; border-left:none; --spin-color:#10b981;">Linear SVM</span>
<span class="insight-box" style="margin-top:0; padding:10px 15px; border-left:none; --spin-color:#10b981;">SQL</span>
<span class="insight-box" style="margin-top:0; padding:10px 15px; border-left:none; --spin-color:#10b981;">Plotly</span>
<span class="insight-box" style="margin-top:0; padding:10px 15px; border-left:none; --spin-color:#10b981;">Streamlit</span>
</div>

</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
<b>Social Engine</b> · Data Vortex 2026 · Round 4
<br>
Data Recovery · SQL Analytics · NLP · Monitoring · Visualization
<br>
Yogesh R Mehta
</div>
""",
    unsafe_allow_html=True
)