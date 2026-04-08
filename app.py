import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Product Recommendation Calculator", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"])

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background-color: #1e293b;
}

/* TEXT COLORS */
label, .stTextInput label, .stTextArea label, .stSelectbox label {
    color: white !important;
    font-weight: 600;
}

/* INPUT BOXES */
.stTextInput>div>div>input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px;
}

/* PLACEHOLDER TEXT */
input::placeholder, textarea::placeholder {
    color: #e0e7ff !important;
}

/* TITLES */
.main-title {
    font-size: 60px;
    font-weight: 800;
    text-align:center;
    color: white;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: white;
    margin-top: 20px;
}

/* METRICS TEXT */
.metric-text {
    color: white;
    font-size: 18px;
    font-weight: 500;
}

/* RESULT TEXT */
.result-good {
    color: #22c55e;
    font-size: 20px;
    font-weight: 600;
}
.result-bad {
    color: #ef4444;
    font-size: 20px;
    font-weight: 600;
}

/* BUTTON */
.stButton>button {
    border-radius: 10px;
    background-color: #2563eb;
    color: white;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">PRODUCT RECOMMENDATION CALCULATOR</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze customer reviews to predict recommendation likelihood</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Review Analysis", "Model Performance", "Dataset"])

# ================= REVIEW ANALYSIS =================
if page == "Review Analysis":

    st.markdown('<p class="section-title">Review Analysis</p>', unsafe_allow_html=True)

    # FILTER
    if "Rating" in df.columns:
        rating = st.slider("⭐ Filter by Rating", 1, 5, (1, 5))
        filtered_df = df[(df["Rating"] >= rating[0]) & (df["Rating"] <= rating[1])]
    else:
        filtered_df = df

    # SEARCH
    search = st.text_input("Search Reviews")

    if search:
        filtered_df = filtered_df[filtered_df["Review Text"].str.contains(search, case=False)]

    reviews = filtered_df["Review Text"].tolist()[:200]

    # INPUT
    example = st.selectbox("Choose Sample Review", [""] + reviews)
    review = st.text_area("Enter Review", value=example, height=120)

    col1, col2 = st.columns(2)
    analyze = col1.button("Analyze")
    clear = col2.button("Clear")

    if clear:
        review = ""

    # ---------------- PREDICTION ----------------
    if analyze and review.strip() != "":
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        prob = model.predict_proba(review_tfidf)[0]

        prob_not = float(prob[0])
        prob_yes = float(prob[1])
        confidence = prob_yes if prediction == 1 else prob_not

        review_length = len(review)
        word_count = len(review.split())

        # ---------------- RESULTS ----------------
        st.markdown('<p class="section-title">Analysis Results</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.markdown(f'<p class="metric-text">Characters: {review_length}</p>', unsafe_allow_html=True)
        col2.markdown(f'<p class="metric-text">Words: {word_count}</p>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        col3.markdown(f'<p class="metric-text">Recommendation Probability: {prob_yes:.2f}</p>', unsafe_allow_html=True)
        col4.markdown(f'<p class="metric-text">Confidence Score: {confidence:.2f}</p>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown('<p class="result-good">✅ Likely Recommended</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result-bad">❌ Likely Not Recommended</p>', unsafe_allow_html=True)

        # ---------------- CONFIDENCE BAR ----------------
        st.markdown('<p class="section-title">Confidence Visualization</p>', unsafe_allow_html=True)

        col_left, col_bar, col_right = st.columns([1, 6, 1])

        col_left.markdown("<p style='color:white;'>0%</p>", unsafe_allow_html=True)

        progress = int(confidence * 100)
        col_bar.progress(progress)

        col_right.markdown(f"<p style='color:white;'>{progress}%</p>", unsafe_allow_html=True)

        # ---------------- PROBABILITY BREAKDOWN ----------------
        st.markdown('<p class="section-title">Probability Breakdown</p>', unsafe_allow_html=True)

        st.markdown("<span style='color:white;'>Recommended</span>", unsafe_allow_html=True)
        st.progress(int(prob_yes * 100))

        st.markdown("<span style='color:white;'>Not Recommended</span>", unsafe_allow_html=True)
        st.progress(int(prob_not * 100))

        # ---------------- GAUGE ----------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Confidence Meter"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "#ef4444"},
                    {'range': [50, 75], 'color': "#f59e0b"},
                    {'range': [75, 100], 'color': "#22c55e"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# ================= MODEL =================
elif page == "Model Performance":
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.87")
    col1.metric("Precision", "0.85")
    col2.metric("Recall", "0.83")
    col2.metric("F1 Score", "0.84")

# ================= DATASET =================
elif page == "Dataset":
    st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(50))

# ---------------- FOOTER ----------------
st.caption("Smart review analysis for product recommendation insights")
