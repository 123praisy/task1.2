import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import time
from wordcloud import WordCloud

st.set_page_config(page_title="Product Recommendation Calculator", layout="wide")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"])

if "cart" not in st.session_state:
    st.session_state.cart = []

if "purchased" not in st.session_state:
    st.session_state.purchased = None

# ================= STYLE =================
st.markdown("""
<style>
.stApp { background-color:#1e3a8a; color:white; }

.main-title {
    font-size:60px; font-weight:800; text-align:center;
}

textarea { background:white !important; color:black !important; }

button {
    background:#2563eb !important;
    color:white !important;
    border-radius:10px !important;
    padding:10px 20px !important;
}

button:hover {
    background:#1d4ed8 !important;
    transform: scale(1.05);
}

/* METRIC CARD */
.metric-card {
    background:white;
    color:black;
    padding:20px;
    border-radius:15px;
    text-align:center;
    transition:0.3s;
}

.metric-card:hover {
    transform:scale(1.05);
    box-shadow:0 0 20px white;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<p class="main-title">PRODUCT RECOMMENDATION CALCULATOR</p>', unsafe_allow_html=True)
st.markdown("---")

# ================= PRODUCT SELECTION =================
st.subheader("🛒 Product Selection")

products = df[['Clothing ID','Class Name']].drop_duplicates().dropna()
options = products.apply(lambda x: f"{x['Clothing ID']} - {x['Class Name']}", axis=1)

selected = st.selectbox("Choose a Product", options)

if st.button("➕ Add to Cart"):
    st.session_state.cart.append(selected)
    st.success("Added to cart!")

st.write("### 🧺 Cart Items")
for item in st.session_state.cart:
    st.write("✔", item)

# ================= ANALYZE =================
if st.button("🔍 Analyze Cart") and st.session_state.cart:

    best_product = None
    best_score = 0

    for item in st.session_state.cart:

        pid = int(item.split(" - ")[0])
        data = df[df["Clothing ID"] == pid]

        text = " ".join(data["Review Text"])

        tfidf = vectorizer.transform([text])
        prob = model.predict_proba(tfidf)[0][1]

        pos = sum(data["Recommended IND"]==1)
        neg = sum(data["Recommended IND"]==0)

        st.markdown(f"## 🛍 {item}")

        c1,c2,c3,c4 = st.columns(4)

        c1.markdown(f"<div class='metric-card'>Recommendation<br><b>{prob:.2f}</b></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'>Positive<br><b>{pos}</b></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'>Negative<br><b>{neg}</b></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'>Total<br><b>{len(data)}</b></div>", unsafe_allow_html=True)

        # ===== PROGRESS BAR =====
        st.write("Confidence Level")

        left, bar, right = st.columns([1,6,1])
        left.write("0%")

        prog = int(prob*100)
        holder = bar.empty()

        for i in range(prog+1):
            holder.progress(i)
            time.sleep(0.005)

        right.write(f"{prog}%")

        # ===== GAUGE =====
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text':"Confidence Meter"},
            gauge={'axis':{'range':[0,100]}}
        ))
        st.plotly_chart(fig)

        # Reason
        if pos>neg:
            st.success("Reason: High positive reviews")
        else:
            st.warning("Reason: Mixed feedback")

        if prob>best_score:
            best_score=prob
            best_product=item

    # ===== FINAL =====
    st.markdown("---")
    st.subheader("🏆 Best Product")

    st.success(f"{best_product} is highly recommended ⭐⭐⭐⭐⭐")

    if st.button("🎉 Purchase Best Product"):
        st.session_state.purchased = best_product
        st.success("Hurray! Product purchased successfully!")

# ================= REVIEW =================
if st.session_state.purchased:

    st.markdown("---")
    st.subheader("📝 Please give your review on the product received")

    review = st.text_area("Write your review")

    if st.button("Submit Review"):

        pid = int(st.session_state.purchased.split(" - ")[0])
        pname = st.session_state.purchased.split(" - ")[1]

        new_data = {
            "Clothing ID": pid,
            "Class Name": pname,
            "Review Text": review,
            "Recommended IND": 1
        }

        df.loc[len(df)] = new_data
        df.to_csv("Womens Clothing E-Commerce Reviews.csv", index=False)

        st.success("Thank you for shopping and your review!")
        st.info("New review added to dataset")
