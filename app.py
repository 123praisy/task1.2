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

# ---------------- SESSION STATE ----------------
if "cart" not in st.session_state:
    st.session_state.cart = []

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp { background-color: #1e3a8a; color: white; }
.main-title { font-size: 60px; font-weight: 800; text-align: center; color: white; }
.subtitle { font-size: 18px; text-align: center; color: #d1d5db; }

textarea { color: black !important; background-color: white !important; }
label { color: white !important; }

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
.metric-value { color: white !important; font-size: 28px; }

.result-good { color: #22c55e; }
.result-bad { color: #ef4444; }
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

    st.subheader("🛒 Product Selection")

    # Create product list
    product_list = df[['Clothing ID', 'Class Name']].drop_duplicates().dropna()

    product_options = product_list.apply(lambda x: f"{x['Clothing ID']} - {x['Class Name']}", axis=1)

    selected_product = st.selectbox("Choose a Product", product_options)

    if st.button("Add to Cart"):
        st.session_state.cart.append(selected_product)
        st.success("Added to cart ✅")

    st.write("🧺 Cart:", st.session_state.cart)

    # ---------------- ANALYZE CART ----------------
    if st.button("Analyze Cart") and st.session_state.cart:

        best_product = None
        best_score = 0

        for product in st.session_state.cart:

            product_id = int(product.split(" - ")[0])

            product_reviews = df[df["Clothing ID"] == product_id]

            if len(product_reviews) == 0:
                continue

            text = " ".join(product_reviews["Review Text"])

            tfidf = vectorizer.transform([text])
            pred = model.predict(tfidf)[0]
            prob = model.predict_proba(tfidf)[0]

            confidence = float(prob[1])

            pos_count = sum(product_reviews["Recommended IND"] == 1)
            neg_count = sum(product_reviews["Recommended IND"] == 0)

            st.markdown(f"### 🛍 Product {product}")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Recommendation %", f"{confidence:.2f}")
            col2.metric("Positive Reviews", pos_count)
            col3.metric("Negative Reviews", neg_count)
            col4.metric("Total Reviews", len(product_reviews))

            # Confidence bar
            progress = int(confidence * 100)
            progress_bar = st.empty()
            for i in range(progress):
                progress_bar.progress(i + 1)
                time.sleep(0.005)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "Confidence Meter"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Reason
            if pos_count > neg_count:
                st.success("Reason: More positive reviews")
            else:
                st.warning("Reason: Mixed or negative reviews")

            if confidence > best_score:
                best_score = confidence
                best_product = product

        # ---------------- FINAL RECOMMENDATION ----------------
        st.markdown("---")
        st.subheader("🏆 Best Product Recommendation")

        st.success(f"{best_product} is highly recommended ⭐⭐⭐⭐⭐")

        if st.button("Purchase Best Product 🎉"):
            st.success("Hurray! Product purchased successfully 🎉")

            st.session_state.purchased = best_product

    # ---------------- USER REVIEW ----------------
    if "purchased" in st.session_state:

        st.markdown("---")
        st.subheader("📝 Give Your Review")

        review_text = st.text_area("Write your review")

        if st.button("Submit Review"):

            new_row = {
                "Clothing ID": int(st.session_state.purchased.split(" - ")[0]),
                "Class Name": st.session_state.purchased.split(" - ")[1],
                "Review Text": review_text,
                "Recommended IND": 1
            }

            df.loc[len(df)] = new_row
            df.to_csv("Womens Clothing E-Commerce Reviews.csv", index=False)

            st.success("✅ Thank you for your review!")
            st.info("New review added to dataset")

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.87")
    col1.metric("Precision", "0.85")
    col2.metric("Recall", "0.83")
    col2.metric("F1 Score", "0.84")

# ================= EDA =================
elif page == "EDA Analysis":

    st.subheader("EDA")

    fig = px.bar(df['Recommended IND'].value_counts())
    st.plotly_chart(fig)

    df['review_length'] = df['Review Text'].apply(len)
    fig = px.histogram(df, x='review_length')
    st.plotly_chart(fig)

    pos_text = " ".join(df[df['Recommended IND']==1]['Review Text'])
    wc = WordCloud().generate(pos_text)
    st.image(wc.to_array())

# ================= DATASET =================
elif page == "Dataset":
    st.dataframe(df.head(50))
