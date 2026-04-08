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
.stApp {
    background-color: #1e3a8a;
    color: white;
}

/* TITLE */
.main-title {
    font-size: 60px;
    font-weight: 800;
    text-align:center;
    color: white;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #d1d5db;
    margin-bottom: 30px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: white;
    margin-top: 25px;
}

/* INPUT LABELS */
label {
    color: white !important;
}

/* BUTTON */
.stButton>button {
    border-radius: 10px;
    background-color: #2563eb;
    color: white;
    font-weight: 600;
}

/* RESULT */
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

/* METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

.metric-title {
    font-size: 16px;
    color: #d1d5db;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">PRODUCT RECOMMENDATION CALCULATOR</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze customer reviews to estimate recommendation likelihood</p>', unsafe_allow_html=True)
st.markdown("---")

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
        filtered_df = filtered_df[
            filtered_df["Review Text"].str.contains(search, case=False)
        ]

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

        st.markdown("---")
        st.markdown('<p class="section-title">Analysis Results</p>', unsafe_allow_html=True)

        # ----------- MODERN METRIC CARDS -----------
        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📝 Characters</div>
            <div class="metric-value">{review_length}</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔤 Words</div>
            <div class="metric-value">{word_count}</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Recommendation</div>
            <div class="metric-value">{prob_yes:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        col4.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Confidence</div>
            <div class="metric-value">{confidence:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown('<p class="result-good">✅ Likely Recommended</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result-bad">❌ Likely Not Recommended</p>', unsafe_allow_html=True)

        # ---------------- CONFIDENCE VISUAL (ANIMATED) ----------------
        st.markdown('<p class="section-title">Confidence Visualization</p>', unsafe_allow_html=True)

        col_left, col_bar, col_right = st.columns([1, 6, 1])

        col_left.markdown("<p style='color:white;'>0%</p>", unsafe_allow_html=True)

        progress_placeholder = col_bar.empty()
        progress_value = int(confidence * 100)

        for i in range(progress_value + 1):
            progress_placeholder.progress(i)
            time.sleep(0.01)

        col_right.markdown(f"<p style='color:white;'>{progress_value}%</p>", unsafe_allow_html=True)

        # ---------------- PROBABILITY ----------------
        st.markdown('<p class="section-title">Probability Breakdown</p>', unsafe_allow_html=True)

        st.markdown("**✅ Recommended**")
        st.progress(int(prob_yes * 100))

        st.markdown("**❌ Not Recommended**")
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
st.markdown("---")
st.caption("Smart review analysis for product recommendation insights")
