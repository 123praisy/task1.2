import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from wordcloud import WordCloud

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Product Recommendation Calculator", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"])

# ================== STYLE ==================
st.markdown("""
<style>
.stApp {
    background-color: #1e3a8a;
    color: white;
}

/* SCROLLING TITLE */
.scroll-title {
    font-size: 70px;
    font-weight: 800;
    white-space: nowrap;
    overflow: hidden;
    display: block;
    animation: scroll-left 10s linear infinite;
    color: white;
}

@keyframes scroll-left {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #d1d5db;
    margin-bottom: 30px;
}

/* METRIC CARD */
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    transition: 0.3s;
}

.metric-card:hover {
    transform: scale(1.05);
    background: rgba(255,255,255,0.2);
    box-shadow: 0px 0px 20px rgba(255,255,255,0.6);
}

.metric-title {
    font-size: 16px;
    color: #d1d5db;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: white !important;
}

/* RESULT */
.result-good { color: #22c55e; font-size: 20px; font-weight: 600; }
.result-bad { color: #ef4444; font-size: 20px; font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="scroll-title">PRODUCT RECOMMENDATION CALCULATOR</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze customer reviews to estimate recommendation likelihood</p>', unsafe_allow_html=True)
st.markdown("---")

# ================= SIDEBAR =================
page = st.sidebar.radio("Navigation", ["Review Analysis", "Model Performance", "EDA Analysis", "Dataset"])

# ================= REVIEW ANALYSIS =================
if page == "Review Analysis":

    review = st.text_area("Enter Review", height=120)

    if st.button("Analyze") and review.strip():

        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        prob = model.predict_proba(review_tfidf)[0]

        prob_yes = float(prob[1])
        confidence = prob_yes if prediction == 1 else float(prob[0])

        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"""<div class="metric-card">
        <div class="metric-title">Characters</div>
        <div class="metric-value">{len(review)}</div></div>""", unsafe_allow_html=True)

        col2.markdown(f"""<div class="metric-card">
        <div class="metric-title">Words</div>
        <div class="metric-value">{len(review.split())}</div></div>""", unsafe_allow_html=True)

        col3.markdown(f"""<div class="metric-card">
        <div class="metric-title">Recommendation</div>
        <div class="metric-value">{prob_yes:.2f}</div></div>""", unsafe_allow_html=True)

        col4.markdown(f"""<div class="metric-card">
        <div class="metric-title">Confidence</div>
        <div class="metric-value">{confidence:.2f}</div></div>""", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown('<p class="result-good">✅ Likely Recommended</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result-bad">❌ Not Recommended</p>', unsafe_allow_html=True)

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    col1, col2 = st.columns(2)

    col1.markdown("""<div class="metric-card">
    <div class="metric-title">Accuracy</div>
    <div class="metric-value">0.87</div></div>""", unsafe_allow_html=True)

    col1.markdown("""<div class="metric-card">
    <div class="metric-title">Precision</div>
    <div class="metric-value">0.85</div></div>""", unsafe_allow_html=True)

    col2.markdown("""<div class="metric-card">
    <div class="metric-title">Recall</div>
    <div class="metric-value">0.83</div></div>""", unsafe_allow_html=True)

    col2.markdown("""<div class="metric-card">
    <div class="metric-title">F1 Score</div>
    <div class="metric-value">0.84</div></div>""", unsafe_allow_html=True)

    df_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Score": [0.87, 0.85, 0.83, 0.84]
    })

    fig = px.bar(df_metrics, x="Metric", y="Score", text="Score")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= EDA ANALYSIS =================
elif page == "EDA Analysis":

    st.subheader("Class Distribution")

    fig = px.bar(df['Recommended IND'].value_counts())
    st.plotly_chart(fig)

    st.subheader("Review Length Distribution")

    df['review_length'] = df['Review Text'].apply(len)
    fig = px.histogram(df, x='review_length')
    st.plotly_chart(fig)

    st.subheader("Word Cloud - Positive Reviews")

    pos_text = " ".join(df[df['Recommended IND']==1]['Review Text'])
    wc = WordCloud().generate(pos_text)
    st.image(wc.to_array())

    st.subheader("Word Cloud - Negative Reviews")

    neg_text = " ".join(df[df['Recommended IND']==0]['Review Text'])
    wc2 = WordCloud().generate(neg_text)
    st.image(wc2.to_array())

# ================= DATASET =================
elif page == "Dataset":
    st.dataframe(df.head(50))
