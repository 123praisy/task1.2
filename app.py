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
    text-align: center;
    color: white;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #d1d5db;
    margin-bottom: 30px;
}

/* INPUT FIX */
textarea {
    color: black !important;
    background-color: white !important;
}

/* LABEL FIX */
label, .stTextArea label {
    color: white !important;
    font-weight: 500;
}

/* BUTTON */
.stButton > button {
    border-radius: 10px;
    background-color: #2563eb;
    color: white;
    font-weight: 600;
}

/* METRIC CARDS */
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
}

/* METRIC TEXT */
.metric-title {
    color: #d1d5db;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: white !important;
}

/* RESULT */
.result-good { color: #22c55e; font-size: 20px; font-weight: 600; }
.result-bad { color: #ef4444; font-size: 20px; font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">PRODUCT RECOMMENDATION CALCULATOR</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze customer reviews to estimate recommendation likelihood</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Review Analysis", "Model Performance", "EDA Analysis", "Dataset"])

# ================= REVIEW ANALYSIS =================
if page == "Review Analysis":

    st.subheader("Review Analysis")

    example = ""
    review = st.text_area("Enter Review", value=example, height=120)

    col1, col2 = st.columns(2)
    analyze = col1.button("Analyze")
    clear = col2.button("Clear")

    if clear:
        review = ""

    if analyze and review.strip() != "":

        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        prob = model.predict_proba(review_tfidf)[0]

        prob_not = float(prob[0])
        prob_yes = float(prob[1])
        confidence = prob_yes if prediction == 1 else prob_not

        st.markdown("---")

        # -------- METRIC CARDS --------
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

        # RESULT
        if prediction == 1:
            st.markdown('<p class="result-good">✅ Likely Recommended</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result-bad">❌ Likely Not Recommended</p>', unsafe_allow_html=True)

        # -------- PROGRESS BAR --------
        st.subheader("Confidence Level")

        progress_placeholder = st.empty()
        progress_value = int(confidence * 100)

        for i in range(progress_value + 1):
            progress_placeholder.progress(i)
            time.sleep(0.01)

        st.write(f"{progress_value}%")

        # -------- GAUGE --------
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

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.subheader("Model Performance")

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
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [0.87, 0.85, 0.83, 0.84]
    })

    fig = px.bar(df_metrics, x="Metric", y="Score", text="Score")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= EDA =================
elif page == "EDA Analysis":

    st.subheader("Class Distribution")
    fig = px.bar(df['Recommended IND'].value_counts())
    st.plotly_chart(fig)

    st.subheader("Review Length Distribution")
    df['review_length'] = df['Review Text'].apply(len)
    fig = px.histogram(df, x='review_length')
    st.plotly_chart(fig)

    st.subheader("Word Cloud - Positive Reviews")
    pos_text = " ".join(df[df['Recommended IND'] == 1]['Review Text'])
    wc = WordCloud().generate(pos_text)
    st.image(wc.to_array())

    st.subheader("Word Cloud - Negative Reviews")
    neg_text = " ".join(df[df['Recommended IND'] == 0]['Review Text'])
    wc2 = WordCloud().generate(neg_text)
    st.image(wc2.to_array())

# ================= DATASET =================
elif page == "Dataset":
    st.dataframe(df.head(50))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Smart review analysis for product recommendation insights")
