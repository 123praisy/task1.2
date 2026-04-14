import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Product Recommendation App", layout="wide")

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

.main-title {
    font-size: 70px;
    font-weight: 800;
    text-align:center;
    color: white;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #d1d5db;
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: white;
    margin-top: 25px;
}

.stButton>button {
    border-radius: 10px;
    background-color: #2563eb;
    color: white;
    font-weight: 600;
}

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

.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #d1d5db;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">Product Recommendation System App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze and predict product recommendation from customer reviews</p>', unsafe_allow_html=True)
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
    search = st.text_input("🔍 Search Reviews")
    if search:
        filtered_df = filtered_df[
            filtered_df["Review Text"].str.contains(search, case=False, na=False)
        ]
        st.markdown(f"Showing results for: **{search}**")

    reviews = filtered_df["Review Text"].tolist()[:200]

    # SAMPLE INPUT
    example = st.selectbox("Choose Sample Review", [""] + reviews)
    review = st.text_area("Enter Review", value=example, height=120)

    col1, col2 = st.columns(2)
    analyze = col1.button("Analyze")
    clear = col2.button("Clear")

    if clear:
        review = ""

    # ---------------- PREDICTION ----------------
    if analyze and review.strip() != "":

        with st.spinner("Analyzing review... 🤖"):
            time.sleep(1.5)

        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        prob = model.predict_proba(review_tfidf)[0]

        prob_not = float(prob[0])
        prob_yes = float(prob[1])
        confidence = prob_yes if prediction == 1 else prob_not

        st.markdown("---")
        st.markdown('<p class="section-title">Analysis Results</p>', unsafe_allow_html=True)

        # METRICS
        col1, col2, col3 = st.columns(3)
        col1.metric("Words", len(review.split()))
        col2.metric("Recommendation %", f"{prob_yes*100:.1f}%")
        col3.metric("Confidence", f"{confidence*100:.1f}%")

        # RESULT
        if prediction == 1:
            st.markdown('<p class="result-good">✅ Likely Recommended</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result-bad">❌ Likely Not Recommended</p>', unsafe_allow_html=True)

        # PROBABILITY BREAKDOWN
        st.markdown('<p class="section-title">Probability Breakdown</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.progress(int(prob_not * 100))
        col1.write(f"❌ Not Recommended: {prob_not*100:.1f}%")

        col2.progress(int(prob_yes * 100))
        col2.write(f"✅ Recommended: {prob_yes*100:.1f}%")

        # CONFIDENCE BAR
        st.markdown('<p class="section-title">Confidence</p>', unsafe_allow_html=True)

        progress = int(confidence * 100)
        st.progress(progress)
        st.write(f"{progress}% confidence")

        # GAUGE
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Confidence Meter"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # MODEL INSIGHT
        st.markdown('<p class="section-title">🧠 Model Insight</p>', unsafe_allow_html=True)
        if prediction == 1:
            st.write("The model detected positive sentiment and supportive words.")
        else:
            st.write("The model detected negative or critical expressions.")

        # TOP WORDS
        st.markdown('<p class="section-title">Top Words</p>', unsafe_allow_html=True)
        words = review.lower().split()
        common_words = Counter(words).most_common(5)
        for word, count in common_words:
            st.write(f"{word} ({count})")

    # ================= NEW USER INPUT SECTION =================
    st.markdown("---")
    st.markdown('<p class="section-title">✨ Test Your Own Review</p>', unsafe_allow_html=True)

    user_review = st.text_area("Write your own review here...", height=120, key="user_input")

    if st.button("Predict My Review 🚀"):
        if user_review.strip() != "":

            with st.spinner("Analyzing your review... 🤖"):
                time.sleep(1.5)

            review_tfidf = vectorizer.transform([user_review])
            prediction = model.predict(review_tfidf)[0]
            prob = model.predict_proba(review_tfidf)[0]

            prob_yes = float(prob[1]) * 100

            if prediction == 1:
                st.success(f"✅ Recommended ({prob_yes:.1f}%)")
            else:
                st.error(f"❌ Not Recommended ({100 - prob_yes:.1f}%)")

        else:
            st.warning("Please enter a review")

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.87")
    col1.metric("Precision", "0.85")
    col2.metric("Recall", "0.83")
    col2.metric("F1 Score", "0.84")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [0.87, 0.85, 0.83, 0.84]
    })

    fig = px.bar(metrics_df, x="Metric", y="Score", text="Score")
    st.plotly_chart(fig, use_container_width=True)

# ================= DATASET =================
elif page == "Dataset":

    st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(50))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Smart AI system for analyzing product reviews")
