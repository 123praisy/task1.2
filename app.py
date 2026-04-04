import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Fashion AI Recommender", layout="wide")

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #ff4b4b;
}
.sub-text {
    font-size: 18px;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">👗 Smart Fashion Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Predict whether a product will be recommended based on customer reviews using Machine Learning.</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "About Model", "Dataset Info"])

# ---------------- HOME PAGE ----------------
if page == "Home":

    st.subheader("📝 Enter Customer Review")

    # Example selector
    example = st.selectbox("💡 Try an example:", [
        "",
        "I absolutely love this dress, perfect fit and amazing quality!",
        "Very bad material, waste of money",
        "Looks nice but size is too small",
        "Comfortable and stylish, would recommend"
    ])

    review = st.text_area("Type your review here:", value=example, height=150)

    col1, col2 = st.columns([1, 1])

    with col1:
        predict_btn = st.button("🔍 Predict")

    with col2:
        clear_btn = st.button("❌ Clear")

    if clear_btn:
        review = ""

    if predict_btn:

        if review.strip() == "":
            st.warning("⚠️ Please enter a review.")
        else:
            # Transform text
            review_tfidf = vectorizer.transform([review])

            # Prediction
            prediction = model.predict(review_tfidf)[0]
            prob = model.predict_proba(review_tfidf)[0]
            confidence = max(prob)

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            # Output display
            if prediction == 1:
                st.success("✅ Recommended")
            else:
                st.error("❌ Not Recommended")

            # Metrics
            col1, col2 = st.columns(2)
            col1.metric("Confidence Score", f"{confidence:.2f}")
            col2.metric("Review Length", len(review))

            # Progress bar
            st.progress(int(confidence * 100))

# ---------------- ABOUT MODEL ----------------
elif page == "About Model":

    st.header("🤖 Model Information")

    st.write("""
    - Model Used: Logistic Regression  
    - Text Processing: TF-IDF Vectorization  
    - Hyperparameter Tuning: GridSearchCV  
    - Evaluation Metric: Accuracy  
    """)

    with st.expander("🔍 How it Works"):
        st.write("""
        The model converts customer reviews into numerical vectors using TF-IDF.  
        It then applies Logistic Regression to classify whether the product is recommended or not.
        """)

# ---------------- DATASET INFO ----------------
elif page == "Dataset Info":

    st.header("📂 Dataset Information")

    st.write("""
    - Dataset: Women’s Clothing E-Commerce Reviews  
    - Features used: Review Text  
    - Target: Recommendation (Yes/No)  
    """)

    with st.expander("📊 Why this dataset?"):
        st.write("""
        This dataset represents real-world customer feedback, making it suitable for building recommendation prediction systems.
        """)

# Footer
st.markdown("---")
st.caption("Built using Machine Learning, Streamlit, and Python")
