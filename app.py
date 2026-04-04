import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Fashion AI Recommender", layout="wide")

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #ff4b4b;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f9f9f9;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
.result-good {
    color: green;
    font-size: 26px;
    font-weight: bold;
}
.result-bad {
    color: red;
    font-size: 26px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">👗 Fashion AI Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Smart prediction of product recommendations using Machine Learning</p>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🤖 Model Insights", "📂 Dataset Info"])

# ---------------- HOME ----------------
if page == "🏠 Home":

    st.markdown("### 📝 Enter Customer Review")

    # Example selector
    example = st.selectbox("💡 Try an example:", [
        "",
        "I absolutely love this dress, perfect fit!",
        "Very bad quality, disappointed",
        "Nice but size is too small",
        "Comfortable and stylish!"
    ])

    review = st.text_area("✍️ Write your review:", value=example, height=150)

    col1, col2 = st.columns([1, 1])

    with col1:
        predict_btn = st.button("🔍 Predict")

    with col2:
        clear_btn = st.button("❌ Clear")

    if clear_btn:
        review = ""

    if predict_btn:

        if review.strip() == "":
            st.warning("⚠️ Please enter a review")
        else:
            review_tfidf = vectorizer.transform([review])

            prediction = model.predict(review_tfidf)[0]
            prob = model.predict_proba(review_tfidf)[0]
            confidence = max(prob)

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            # Styled result card
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown('<p class="result-good">✅ Recommended 😊</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="result-bad">❌ Not Recommended 😞</p>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                col1.metric("🎯 Confidence", f"{confidence:.2f}")
                col2.metric("📏 Review Length", len(review))

                st.progress(int(confidence * 100))

                st.markdown('</div>', unsafe_allow_html=True)

            # Explanation expanders
            with st.expander("📘 How is Confidence Calculated?"):
                st.write("""
                The model outputs probabilities for each class using Logistic Regression.
                The confidence score is the highest probability value.
                """)

            with st.expander("📏 How is Review Length Calculated?"):
                st.write("""
                Review length is the number of characters in the input text.
                """)

# ---------------- MODEL INSIGHTS ----------------
elif page == "🤖 Model Insights":

    st.header("🤖 Model Insights")

    st.markdown("### 🧠 Model Overview")
    st.write("""
    - Model: Logistic Regression  
    - Text Processing: TF-IDF Vectorization  
    - Tuning: GridSearchCV  
    """)

    st.markdown("### 📊 Performance Metrics")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.87")
    col1.metric("Precision", "0.85")
    col2.metric("Recall", "0.83")
    col2.metric("F1 Score", "0.84")

    with st.expander("📘 What do these metrics mean?"):
        st.write("""
        Accuracy: Overall correctness  
        Precision: Correct positive predictions  
        Recall: Coverage of actual positives  
        F1 Score: Balance between precision & recall  
        """)

# ---------------- DATASET ----------------
elif page == "📂 Dataset Info":

    st.header("📂 Dataset Info")

    st.write("""
    Dataset: Women’s Clothing E-Commerce Reviews  
    Target: Recommendation (Yes/No)  
    Feature: Review Text  
    """)

    st.info("This dataset contains real customer reviews used to train the ML model.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit | Machine Learning Project")
