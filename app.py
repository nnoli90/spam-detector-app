# ===============================
# SPAM DETECTION WEB APP
# ===============================

import streamlit as st
import joblib
import re
import string
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Text cleaning function (must match training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Load models
model, vectorizer = load_model()

# ===============================
# UI COMPONENTS
# ===============================

# Header
st.title("📧 SMS Spam Detector")
st.markdown("""
    <p style='font-size: 18px;'>
    Enter a message below to check if it's <strong style='color:#e74c3c'>SPAM</strong> or 
    <strong style='color:#2ecc71'>HAM (Not Spam)</strong>.
    </p>
""", unsafe_allow_html=True)

# Input section
st.subheader("📝 Enter Your Message")

# Text input
user_input = st.text_area(
    "Type your message here:",
    height=120,
    placeholder="Example: CONGRATULATIONS! You've won a $1000 Amazon gift card..."
)

# Create two columns for buttons
col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("🔍 Predict", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

# Clear functionality
if clear_button:
    user_input = ""
    st.rerun()

# ===============================
# PREDICTION
# ===============================

if predict_button and user_input:
    with st.spinner("Analyzing message..."):
        # Clean and transform
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0][prediction]
        
        # Display result
        st.markdown("---")
        st.subheader("🔎 Result")
        
        if prediction == 1:  # SPAM
            st.error(f"🚨 **SPAM DETECTED!**")
            st.markdown(f"""
                <div style='background-color:#fde8e8; padding:15px; border-radius:10px;'>
                    <p style='margin:0; color:#c0392b; font-size:16px;'>
                        ⚠️ This message appears to be <strong>SPAM</strong> with {confidence:.1%} confidence.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:  # HAM
            st.success(f"✅ **HAM (Not Spam)**")
            st.markdown(f"""
                <div style='background-color:#e8f8e8; padding:15px; border-radius:10px;'>
                    <p style='margin:0; color:#27ae60; font-size:16px;'>
                        ✓ This appears to be a legitimate message with {confidence:.1%} confidence.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Show cleaned text (optional)
        with st.expander("🔧 See processed text"):
            st.write(f"**Cleaned:** {cleaned}")
            st.write(f"**Confidence Score:** {confidence:.2%}")

elif predict_button and not user_input:
    st.warning("⚠️ Please enter a message to classify.")

# ===============================
# SIDEBAR - INFO
# ===============================

with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | **Model** | Naive Bayes |
    | **Accuracy** | 96.89% |
    | **Precision** | 97.50% |
    | **Recall** | 82.98% |
    | **F1-Score** | 89.66% |
    """)
    
    st.header("📝 Example Messages")
    st.markdown("""
    **Try these examples:**
    
    *Spam:*
    > "WINNER! You've won a free iPhone. Click here to claim."
    
    *Ham:*
    > "Hey, are we still meeting for lunch at 2pm?"
    """)
    
    st.header("📧 About")
    st.markdown("""
    This app uses **TF-IDF vectorization** and a **Multinomial Naive Bayes** classifier to detect spam messages in real-time.
    """)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>Spam Detection Model | Built with Streamlit</p>",
    unsafe_allow_html=True
)