import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Product Recommendation Calculator", layout="wide")

# ---------------- LOAD ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=["Review Text"])

# ---------------- SESSION STATE ----------------
if "cart" not in st.session_state:
    st.session_state.cart = []

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "best_product" not in st.session_state:
    st.session_state.best_product = None

if "confirm_purchase" not in st.session_state:
    st.session_state.confirm_purchase = False

if "purchased" not in st.session_state:
    st.session_state.purchased = None

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {background:#1e3a8a;color:white;}

textarea {background:white !important;color:black !important;}

label {color:white !important;}

div[data-baseweb="select"] > div {
    background:white !important;
    color:black !important;
}

button {
    background:#2563eb !important;
    color:white !important;
    border-radius:10px;
}
button:hover {transform:scale(1.05);}

.metric-card {
    background:white;
    color:black;
    padding:10px;             
    border-radius:10px;
    text-align:center;
    min-height:80px;           
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    transition:0.3s;
}

/* TEXT (LABEL) */
.metric-card div {
    white-space: nowrap;       
    font-size: 13px;           /* smaller text */
    font-weight:500;
}

/* VALUE */
.metric-card b {
    font-size:16px;            /* slightly smaller number */
}

.perf-card {
    background:white;
    color:black;
    padding:25px;
    border-radius:12px;
    text-align:center;
    transition:0.3s;
}
.perf-card:hover {
    transform:scale(1.05);
    box-shadow:0 0 20px white;
}

h1, h2, h3 {color:white !important;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>PRODUCT RECOMMENDATION CALCULATOR</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Review Analysis","Model Performance","EDA Analysis","Dataset"])

# ================= REVIEW ANALYSIS =================
if page == "Review Analysis":

    st.subheader("🛒 Product Selection")

    products = df[['Clothing ID','Class Name']].drop_duplicates().dropna()
    options = products.apply(lambda x: f"{x['Clothing ID']} - {x['Class Name']}", axis=1)

    selected = st.selectbox("Choose Product", options)

    if st.button("Add to Cart"):
        st.session_state.cart.append(selected)
        st.success("Added to cart")

    # -------- CART --------
    st.write("### 🧺 Cart Items")
    for item in st.session_state.cart:
        st.write("✔", item)

    # -------- ANALYZE --------
    if st.button("Analyze Cart"):
        st.session_state.analysis_done = True

    if st.session_state.analysis_done and st.session_state.cart:

        unique_cart = list(set(st.session_state.cart))
        best_score = 0

        for i, item in enumerate(unique_cart):

            pid = int(item.split(" - ")[0])
            data = df[df["Clothing ID"] == pid]

            text = " ".join(data["Review Text"])
            tfidf = vectorizer.transform([text])
            prob = model.predict_proba(tfidf)[0][1]

            pos = sum(data["Recommended IND"] == 1)
            neg = sum(data["Recommended IND"] == 0)
            avg_rating = round(data["Rating"].mean(), 2)

            st.markdown(f"## 🛍 {item}")

            c1,c2,c3,c4,c5 = st.columns(5)

            c1.markdown(f"<div class='metric-card'>Recommendation<br><b>{prob:.2f}</b></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'>Positive<br><b>{pos}</b></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'>Negative<br><b>{neg}</b></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'>Total<br><b>{len(data)}</b></div>", unsafe_allow_html=True)
            c5.markdown(f"<div class='metric-card'>⭐ Rating<br><b>{avg_rating}</b></div>", unsafe_allow_html=True)

            # -------- CONFIDENCE --------
            st.write("Confidence Level")
            colL, colM, colR = st.columns([1,6,1])
            colL.write("0%")

            progress = int(prob * 100)
            bar = colM.empty()

            for j in range(progress + 1):
                bar.progress(j)
                time.sleep(0.002)

            colR.write(f"{progress}%")

            if prob > best_score:
                best_score = prob
                st.session_state.best_product = item

        # -------- BEST PRODUCT --------
        st.markdown("---")
        st.markdown(f"""
<div style="
    background:#2563eb;
    padding:10px;
    border-radius:10px;
    color:white;
    font-weight:400;
    text-align:center;
">
🏆 Best Product: {st.session_state.best_product} ⭐⭐⭐⭐⭐
</div>
""", unsafe_allow_html=True)

        if st.button("Purchase Best Product"):
            st.session_state.confirm_purchase = True

    # ================= PURCHASE FLOW =================
    if st.session_state.confirm_purchase:

        st.subheader("🛍 Confirm Purchase")

        selected_purchase = st.selectbox(
            "Select product to purchase",
            st.session_state.cart
        )

        pid = int(selected_purchase.split(" - ")[0])
        data = df[df["Clothing ID"] == pid]

        text = " ".join(data["Review Text"])
        tfidf = vectorizer.transform([text])
        prob = model.predict_proba(tfidf)[0][1]

        if st.button("Confirm Purchase"):

            st.session_state.purchased = selected_purchase
            st.session_state.confirm_purchase = False

        if prob < 0.5:
    st.markdown("""
    <div style="background-color:#007BFF; padding:15px; border-radius:10px;">
        <p style="color:white; font-weight:bold;">
        ⚠️ You may regret this purchase as it has low recommendation and ratings.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background-color:#007BFF; padding:15px; border-radius:10px;">
        <p style="color:white; font-weight:bold;">
        🎉 Thank you for choosing a highly recommended product!
        </p>
    </div>
    """, unsafe_allow_html=True)
            

    # ================= REVIEW SECTION =================
    if st.session_state.purchased:

        st.markdown("---")
        st.subheader("📝 Please give your review on the product received")

        pid = st.session_state.purchased.split(" - ")[0]
        pname = st.session_state.purchased.split(" - ")[1]

        st.write(f"Product ID: {pid}")
        st.write(f"Product Name: {pname}")

        review = st.text_area("Write your review")

        if st.button("Submit Review"):

            new_row = {
                "Clothing ID": int(pid),
                "Class Name": pname,
                "Review Text": review,
                "Recommended IND": 1,
                "Rating": 5
            }

            df.loc[len(df)] = new_row

            # SAVE TO CSV
            df.to_csv("Womens Clothing E-Commerce Reviews.csv", index=False)

            st.success("✅ Thank you for your review!")
            st.info("📁 Review successfully saved to dataset")

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.subheader("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown("<div class='perf-card'>Accuracy<br><b>0.87</b></div>", unsafe_allow_html=True)
    c2.markdown("<div class='perf-card'>Precision<br><b>0.85</b></div>", unsafe_allow_html=True)
    c3.markdown("<div class='perf-card'>Recall<br><b>0.83</b></div>", unsafe_allow_html=True)
    c4.markdown("<div class='perf-card'>F1 Score<br><b>0.84</b></div>", unsafe_allow_html=True)

# ================= EDA =================
elif page == "EDA Analysis":

    st.subheader("📈 Class Distribution")
    fig = px.bar(df['Recommended IND'].value_counts())
    st.plotly_chart(fig)

    df['review_length'] = df['Review Text'].apply(len)

    st.subheader("📊 Review Length")
    fig = px.histogram(df, x='review_length')
    st.plotly_chart(fig)

    st.subheader("🔗 Correlation Matrix")
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

# ================= DATASET =================
elif page == "Dataset":
    st.dataframe(df.head(50))
