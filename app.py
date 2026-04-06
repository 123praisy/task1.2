import streamlit as st
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fashion AI Recommender", layout="wide")

# ---------------- LOAD MODEL ----------------
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

# ---------------- GAUGE FUNCTION ----------------
def show_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'thickness': 0.3},
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
st.markdown('<p class="subtitle">Smart prediction of product recommendations using Machine Learning</p>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🤖 Model Insights", "📂 Dataset Info"])

# ================= HOME =================
if page == "🏠 Home":

    st.markdown("### 📝 Enter Customer Review")

    example = st.selectbox("💡 Try an example:", [
        "",
        "I absolutely love this dress, perfect fit!",
        "Very bad quality, disappointed",
        "Nice but size is too small",
        "Comfortable and stylish!"
    ])

    review = st.text_area("✍️ Write your review:", value=example, height=150)

    col1, col2 = st.columns(2)

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
            # Transform text
            review_tfidf = vectorizer.transform([review])

            # Prediction
            prediction = model.predict(review_tfidf)[0]
            prob = model.predict_proba(review_tfidf)[0]

            prob_not_rec = prob[0]
            prob_rec = prob[1]
            confidence = max(prob)

            # Review length
            review_length = len(review)
            word_count = len(review.split())

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            # Result card
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown('<p class="result-good">✅ Recommended 😊</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="result-bad">❌ Not Recommended 😞</p>', unsafe_allow_html=True)

                # Metrics
                col1, col2 = st.columns(2)
                col1.metric("🎯 Confidence", f"{confidence:.2f}")
                col2.metric("📏 Characters", review_length)

                col3, col4 = st.columns(2)
                col3.metric("📝 Words", word_count)
                col4.metric("📊 Probability (Recommended)", f"{prob_rec:.2f}")

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

            # Explanation sections
            with st.expander("📘 How is Confidence Calculated?"):
                st.write(f"""
                The model predicts probabilities for two classes:

                - Not Recommended (0): {prob_not_rec:.2f}  
                - Recommended (1): {prob_rec:.2f}

                The predicted class is the one with the higher probability.

                Confidence Score = max(probabilities)

                In this case:
                Confidence = {confidence:.2f}
                """)

            with st.expander("📏 How is Review Length Calculated?"):
                st.write(f"""
                Characters = len(review) → {review_length}

                Words = len(review.split()) → {word_count}

                This helps understand how detailed the review is.
                """)

# ================= MODEL =================
elif page == "🤖 Model Insights":

    st.header("🤖 Model Insights")

    st.markdown("### 🧠 Model Overview")
    st.write("""
    - Model: Logistic Regression  
    - Text Processing: TF-IDF Vectorization  
    - Hyperparameter Tuning: GridSearchCV  
    """)

    st.markdown("### 📊 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.87")
    col1.metric("Precision", "0.85")
    col2.metric("Recall", "0.83")
    col2.metric("F1 Score", "0.84")

    with st.expander("📘 What do these metrics mean?"):
        st.write("""
        Accuracy: Overall correctness of the model  
        Precision: Correct positive predictions  
        Recall: Coverage of actual positives  
        F1 Score: Balance between precision and recall  
        """)

# ================= DATASET =================
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
