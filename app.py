import streamlit as st
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fashion AI Recommender", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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
st.markdown('<p class="subtitle">Predict product recommendation using ML</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🤖 Model Insights", "📂 Dataset Info"])

# ================= HOME =================
if page == "🏠 Home":

    st.markdown("### 📝 Enter Customer Review")

    example = st.selectbox("💡 Try an example:", [
        "",
        "I absolutely love this dress!",
        "Worst product ever",
        "Nice but size is small",
        "Very comfortable and stylish"
    ])

    review = st.text_area("✍️ Write your review:", value=example, height=150)

    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔍 Predict")
    clear_btn = col2.button("❌ Clear")

    if clear_btn:
        review = ""

    if predict_btn:

        if review.strip() == "":
            st.warning("⚠️ Please enter a review")
        else:
            # 🔥 IMPORTANT: Always recompute
            review_tfidf = vectorizer.transform([review])

            prediction = model.predict(review_tfidf)[0]
            prob = model.predict_proba(review_tfidf)[0]

            prob_not_rec = float(prob[0])
            prob_rec = float(prob[1])

            # 🔥 Correct confidence based on prediction
            if prediction == 1:
                confidence = prob_rec
            else:
                confidence = prob_not_rec

            # Length
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

            # 🔥 Show both probabilities
            st.markdown("### 📊 Prediction Breakdown")
            col1, col2 = st.columns(2)
            col1.metric("❌ Not Recommended", f"{prob_not_rec:.2f}")
            col2.metric("✅ Recommended", f"{prob_rec:.2f}")

            # 🔥 Gauge
            st.markdown("### 🎯 Confidence Meter")
            show_confidence_gauge(confidence)

            # 🔥 Confidence message
            if confidence > 0.75:
                st.success("🔥 High confidence prediction")
            elif confidence > 0.5:
                st.info("⚖️ Moderate confidence")
            else:
                st.warning("⚠️ Low confidence prediction")

            # 🔥 Explanation
            with st.expander("📘 How is Confidence Calculated?"):
                st.write(f"""
                Model probabilities:
                - Not Recommended: {prob_not_rec:.2f}
                - Recommended: {prob_rec:.2f}

                The predicted class determines confidence.

                Confidence = probability of predicted class

                → {confidence:.2f}
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

    st.write("""
    Dataset: Women's Clothing Reviews  
    Target: Recommendation  
    Feature: Review Text  
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit")
