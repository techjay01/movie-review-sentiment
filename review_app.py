
import streamlit as st
import joblib
import re

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = joblib.load("good_bad_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Project Details")
st.sidebar.markdown("""
**Course:** CSC 401  
**Project:** Movie Review Sentiment Analysis  
**Model:** Logistic Regression  
**Dataset:** IMDb Mini  
**Features:** TF-IDF  
**Output:** Good / Bad Review  
""")

st.sidebar.markdown("---")
st.sidebar.info("Enter a movie review on the main page to analyze sentiment.")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("<h1 style='text-align: center;'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center;'>Predict whether a movie review is <b>GOOD</b> or <b>BAD</b> using Machine Learning</p>",
    unsafe_allow_html=True
)

st.markdown("---")

review = st.text_area(
    "✍️ Enter a movie review:",
    height=150,
    placeholder="Example: The movie was amazing with great acting and storyline..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Analyze Review"):
    if review.strip():
        cleaned = preprocess_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            confidence = probability[1]
            st.success("✅ **GOOD REVIEW**")
        else:
            confidence = probability[0]
            st.error("❌ **BAD REVIEW**")

        st.write(f"**Confidence Level:** {confidence * 100:.2f}%")
        st.progress(confidence)

    else:
        st.warning("⚠️ Please enter a review before clicking analyze.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>Developed by Group 12 | CSC 401</p>",
    unsafe_allow_html=True
)
