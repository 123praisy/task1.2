import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Product Recommendation Calculator", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
    df.columns = df.columns.str.strip()
    text_col = "Review Text" if "Review Text" in df.columns else "review"
    df = df.dropna(subset=[text_col])
    return df, text_col

df, text_col = load_data()

# ---------------- DARK UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
.main-title {
    font-size: 60px;
    font-weight: 800;
    text-align:center;
    color: white;
}
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #9ca3af;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #e5e7eb;
}
.card {
    padding: 25px;
    border-radius: 15px;
    background: #1f2937;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.5);
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
        filtered_df = filtered_df[filtered_df[text_col].str.contains(search, case=False)]

    reviews = filtered_df[text_col].tolist()[:200]

    # INPUT
    example = st.selectbox("Choose Sample Review", [""] + reviews)
    review = st.text_area("Enter Review", value=example, height=120)

    col1, col2 = st.columns(2)
    analyze = col1.button("Analyze")
    clear = col2.button("Clear")

    if clear:
        st.rerun()

    # ---------------- PREDICTION ----------------
    if analyze:
        if review.strip() == "":
            st.warning("Please enter a review")
        else:
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

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                col1.metric("Characters", review_length)
                col2.metric("Words", word_count)

                col3, col4 = st.columns(2)
                col3.metric("Recommendation Probability", f"{prob_yes:.2f}")
                col4.metric("Confidence Score", f"{confidence:.2f}")

                st.markdown("<br>", unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown('<p class="result-good">✅ Likely Recommended</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="result-bad">❌ Likely Not Recommended</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # ---------------- CONFIDENCE BAR ----------------
            st.markdown('<p class="section-title">📊 Confidence</p>', unsafe_allow_html=True)

            progress_bar = st.progress(0)
            for i in range(int(confidence * 100) + 1):
                time.sleep(0.005)
                progress_bar.progress(i)

            st.markdown(f"### Confidence: {confidence*100:.1f}%")

            # ---------------- PROBABILITY ----------------
            st.markdown('<p class="section-title">📊 Probability Breakdown</p>', unsafe_allow_html=True)

            st.markdown("**Recommended**")
            st.progress(int(prob_yes * 100))

            st.markdown("**Not Recommended**")
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

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)

    try:
        metrics = joblib.load("metrics.pkl")

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        col1.metric("Precision", f"{metrics['precision']:.2f}")
        col2.metric("Recall", f"{metrics['recall']:.2f}")
        col2.metric("F1 Score", f"{metrics['f1']:.2f}")

    except:
        st.warning("Metrics file not found.")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", "N/A")
        col1.metric("Precision", "N/A")
        col2.metric("Recall", "N/A")
        col2.metric("F1 Score", "N/A")

# ================= DATASET =================
elif page == "Dataset":
    st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(50))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Smart review analysis for product recommendation insights")
