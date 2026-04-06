# ---------------- CONFIDENCE VISUAL ----------------
st.markdown('<p class="section-title">📊 Confidence Visualization</p>', unsafe_allow_html=True)

# Layout: Left 0% | Bar | Right %
col1, col2, col3 = st.columns([1, 6, 1])
col1.markdown("**0%**")  # Left start
progress_placeholder = col2.empty()
percent_placeholder = col3.empty()  # Right end

# Animate progress bar
for i in range(int(confidence * 100) + 1):
    time.sleep(0.01)
    progress_placeholder.progress(i)
    percent_placeholder.markdown(f"**{i}%**")

# Final confidence status
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
