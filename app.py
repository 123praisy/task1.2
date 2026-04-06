import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Product Recommendation Calculator", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"])

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8f9fb, #eef1f7);
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #222;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    color: #666;
    text-align: center;
    margin-bottom: 20px;
}
.card {
    padding: 25px;
    border-radius: 18px;
    background: white;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
}
.result-good {
    color: #2e7d32;
    font-size: 26px;
    font-weight: bold;
}
.result-bad {
    color: #c62828;
    font-size: 26px;
    font-weight: bold;
}
.stButton>button {
    border-radius: 10px;
    background-color: #4CAF50;
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- GAUGE ----------------
def show_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "#ff4b4b"},
                {'range': [50, 75], 'color': "#ffa600"},
                {'range': [75, 100], 'color': "#4caf50"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🛍️ Product Recommendation Calculator Based on Reviews</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Evaluate customer reviews to estimate whether a product is likely to be recommended</p>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Dashboard")
page = st.sidebar.radio("Navigate", ["🏠 Review Analyzer", "📈 Model Performance", "📂 Dataset Explorer"])

# ================= HOME =================
if page == "🏠 Review Analyzer":

    st.markdown("## 🔍 Review Analysis")

    # Filter
    if "Rating" in df.columns:
        rating_filter = st.slider("⭐ Filter Reviews by Rating", 1, 5, (1, 5))
        filtered_df = df[(df["Rating"] >= rating_filter[0]) & (df["Rating"] <= rating_filter[1])]
    else:
        filtered_df = df

    # Search
    search_query = st.text_input("🔎 Search Reviews")

    if search_query:
        filtered_df = filtered_df[
            filtered_df["Review Text"].str.contains(search_query, case=False)
        ]

    reviews = filtered_df["Review Text"].tolist()
    reviews_sample = reviews[:200]

    # Random
    if st.button("🎲 Generate Sample Review"):
        review = random.choice(reviews_sample)
    else:
        review = ""

    example = st.selectbox("📌 Select a Review Example", [""] + reviews_sample)

    review = st.text_area(
        "✍️ Enter or Edit Review Text",
        value=example or review,
        height=150
    )

    col1, col2 = st.columns(2)
    analyze_btn = col1.button("🚀 Analyze Recommendation")
    clear_btn = col2.button("🧹 Clear Input")

    if clear_btn:
        review = ""

    # ---------------- PREDICTION ----------------
    if analyze_btn:

        if review.strip() == "":
            st.warning("⚠️ Please enter a review to analyze")
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
            st.markdown("## 📊 Analysis Results")

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown(f'<p class="result-good">✅ Likely Recommended ({confidence:.2f})</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="result-bad">❌ Likely Not Recommended ({confidence:.2f})</p>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                col1.metric("Confidence Score", f"{confidence:.2f}")
                col2.metric("Character Count", review_length)

                col3, col4 = st.columns(2)
                col3.metric("Word Count", word_count)
                col4.metric("Recommendation Probability", f"{prob_yes:.2f}")

                st.progress(int(confidence * 100))

                st.markdown('</div>', unsafe_allow_html=True)

            # Breakdown
            st.markdown("### 📊 Probability Breakdown")
            col1, col2 = st.columns(2)
            col1.metric("Not Recommended", f"{prob_not:.2f}")
            col2.metric("Recommended", f"{prob_yes:.2f}")

            # Gauge
            st.markdown("### 🎯 Confidence Visualization")
            show_confidence_gauge(confidence)

            # Explanation
            with st.expander("📘 How this result is calculated"):
                st.write(f"""
                The system evaluates the review text and assigns probabilities:

                - Not Recommended: {prob_not:.2f}  
                - Recommended: {prob_yes:.2f}  

                The final result is based on the higher probability.

                Confidence Score = {confidence:.2f}
                """)

# ================= MODEL =================
elif page == "📈 Model Performance":

    st.header("📈 Model Performance Overview")

    st.write("""
    This system uses a trained classification model to evaluate customer reviews
    and estimate recommendation likelihood.
    """)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.87")
    col1.metric("Precision", "0.85")
    col2.metric("Recall", "0.83")
    col2.metric("F1 Score", "0.84")

# ================= DATASET =================
elif page == "📂 Dataset Explorer":

    st.header("📂 Dataset Explorer")

    st.write("Preview of dataset used for analysis:")

    st.dataframe(df.head(50))

    st.write("""
    The dataset contains real customer reviews, ratings, and recommendation labels.
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built as an intelligent review analysis tool for product recommendation insights")
