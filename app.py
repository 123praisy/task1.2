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
    background-color: #f5f7fb;
}
/* MAIN TITLE */
.main-title {
    font-size: 60px;
    font-weight: 800;
    text-align:center;
    color: 'black';
}
/* SUBTITLE */
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}
/* SECTION TITLE */
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #374151;
    margin-top: 25px;
}
/* CARD */
.card {
    padding: 25px;
    border-radius: 15px;
    background: white;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}
/* RESULT */
.result-good {
    color: #16a34a;
    font-size: 20px;
    font-weight: 600;
}
.result-bad {
    color: #dc2626;
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
st.markdown('<p class="main-title"> PRODUCT RECOMMENDATION CALCULATOR BASED ON REVIEWS</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze customer reviews to estimate product recommendation likelihood</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Review Analysis", "Model Performance", "Dataset"])

# ================= HOME =================
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

            # RESULT TEXT
            if prediction == 1:
                st.markdown(f'<p class="result-good">✅ Likely Recommended</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="result-bad">❌ Likely Not Recommended</p>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- CONFIDENCE VISUAL ----------------
        st.markdown('<p class="section-title">📊 Confidence Visualization</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 6, 1])
        col1.markdown("**0%**")
        progress_placeholder = col2.empty()
        percent_placeholder = col3.empty()

        for i in range(int(confidence * 100) + 1):
            time.sleep(0.01)
            progress_placeholder.progress(i)
            percent_placeholder.markdown(f"**{i}%**")

        if confidence > 0.75:
            st.success(f"🔥 High Confidence ({confidence*100:.0f}%)")
        elif confidence > 0.5:
            st.info(f"⚖️ Moderate Confidence ({confidence*100:.0f}%)")
        else:
            st.warning(f"⚠️ Low Confidence ({confidence*100:.0f}%)")

        # ---------------- PROBABILITY BREAKDOWN ----------------
        st.markdown('<p class="section-title">📊 Probability Breakdown</p>', unsafe_allow_html=True)

        # Recommended
        st.markdown("**✅ Recommended**")
        col1, col2 = st.columns([8, 2])
        col1.progress(int(prob_yes * 100))
        col2.markdown(f"**{prob_yes*100:.0f}%**")

        # Not Recommended
        st.markdown("**❌ Not Recommended**")
        col3, col4 = st.columns([8, 2])
        col3.progress(int(prob_not * 100))
        col4.markdown(f"**{prob_not*100:.0f}%**")

        st.caption("Left: Not Recommended | Right: Recommended")

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
