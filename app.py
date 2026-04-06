import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fashion AI Recommender", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"])

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 700; color: #ff4b4b;}
.subtitle {font-size: 18px; color: #555;}
.card {padding: 20px; border-radius: 15px; background-color: #f9f9f9;
       box-shadow: 0px 4px 10px rgba(0,0,0,0.05);}
.result-good {color: green; font-size: 26px; font-weight: bold;}
.result-bad {color: red; font-size: 26px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ---------------- GAUGE ----------------
def show_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Level"},
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
st.markdown('<p class="main-title">👗 Fashion AI Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict product recommendations using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🤖 Model Insights", "📂 Dataset Info"])

# ================= HOME =================
if page == "🏠 Home":

    st.markdown("### 🔍 Explore Reviews")

    # ⭐ Rating filter
    if "Rating" in df.columns:
        rating_filter = st.slider("⭐ Filter by Rating", 1, 5, (1, 5))
        filtered_df = df[(df["Rating"] >= rating_filter[0]) & (df["Rating"] <= rating_filter[1])]
    else:
        filtered_df = df

    # 🔍 Search reviews
    search_query = st.text_input("🔍 Search keyword in reviews")

    if search_query:
        filtered_df = filtered_df[filtered_df["Review Text"].str.contains(search_query, case=False)]

    reviews_list = filtered_df["Review Text"].tolist()

    # Limit size
    reviews_sample = reviews_list[:200]

    # 🎲 Random review
    if st.button("🎲 Random Review"):
        review = random.choice(reviews_sample)
    else:
        review = ""

    # Selectbox
    example = st.selectbox("💡 Choose a review:", [""] + reviews_sample)

    # Text input
    review = st.text_area("✍️ Or write your own review:", value=example or review, height=150)

    # Buttons
    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔍 Predict")
    clear_btn = col2.button("❌ Clear")

    if clear_btn:
        review = ""

    # ---------------- PREDICTION ----------------
    if predict_btn:

        if review.strip() == "":
            st.warning("⚠️ Please enter a review")
        else:
            review_tfidf = vectorizer.transform([review])

            prediction = model.predict(review_tfidf)[0]
            prob = model.predict_proba(review_tfidf)[0]

            prob_not_rec = float(prob[0])
            prob_rec = float(prob[1])

            confidence = prob_rec if prediction == 1 else prob_not_rec

            review_length = len(review)
            word_count = len(review.split())

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown(f'<p class="result-good">✅ Recommended ({confidence:.2f})</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="result-bad">❌ Not Recommended ({confidence:.2f})</p>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                col1.metric("🎯 Confidence", f"{confidence:.2f}")
                col2.metric("📏 Characters", review_length)

                col3, col4 = st.columns(2)
                col3.metric("📝 Words", word_count)
                col4.metric("📊 Recommended Prob", f"{prob_rec:.2f}")

                st.progress(int(confidence * 100))

                st.markdown('</div>', unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("### 📊 Prediction Breakdown")
            col1, col2 = st.columns(2)
            col1.metric("❌ Not Recommended", f"{prob_not_rec:.2f}")
            col2.metric("✅ Recommended", f"{prob_rec:.2f}")

            # Gauge
            st.markdown("### 🎯 Confidence Meter")
            show_confidence_gauge(confidence)

            # Confidence level message
            if confidence > 0.75:
                st.success("🔥 High confidence prediction")
            elif confidence > 0.5:
                st.info("⚖️ Moderate confidence")
            else:
                st.warning("⚠️ Low confidence prediction")

            # Explanation
            with st.expander("📘 How is Confidence Calculated?"):
                st.write(f"""
                - Not Recommended: {prob_not_rec:.2f}  
                - Recommended: {prob_rec:.2f}  

                Confidence = probability of predicted class → {confidence:.2f}
                """)

            with st.expander("📏 Review Length Calculation"):
                st.write(f"""
                Characters = {review_length}  
                Words = {word_count}
                """)

# ================= MODEL =================
elif page == "🤖 Model Insights":

    st.header("🤖 Model Insights")

    st.write("""
    - Model: Logistic Regression  
    - Text Processing: TF-IDF  
    - Tuning: GridSearchCV  
    """)

    st.markdown("### 📊 Performance")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.87")
    col1.metric("Precision", "0.85")
    col2.metric("Recall", "0.83")
    col2.metric("F1 Score", "0.84")

# ================= DATASET =================
elif page == "📂 Dataset Info":

    st.header("📂 Dataset Info")

    st.write("### 📊 Sample Dataset")
    st.dataframe(df.head(50))

    st.write("""
    Dataset: Women’s Clothing E-Commerce Reviews  
    Target: Recommendation  
    Feature: Review Text  
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit | ML Project")
