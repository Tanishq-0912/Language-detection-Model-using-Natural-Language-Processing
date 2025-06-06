import streamlit as st
import joblib
import os

# ========== Step 1: Install gdown ==========
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

# ========== Step 2: Download model and vectorizer from Google Drive ==========

model_file = "NLP_model.pkl"
vectorizer_file = "vectorizer.pkl"

# Replace with your actual Google Drive File IDs
model_file_id = "1-tdIZCB2IbtHpyt4Jqlq8H1QPHwhH2x4"
vectorizer_file_id = "PUT_VECTOR_ID_HERE"  # 🔁 Replace this!

if not os.path.exists(model_file):
    gdown.download(id=model_file_id, output=model_file, quiet=False)

if not os.path.exists(vectorizer_file):
    gdown.download(id=vectorizer_file_id, output=vectorizer_file, quiet=False)

# ========== Step 3: Load model ==========
model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)

# ========== Step 4: Streamlit UI ==========
st.set_page_config(page_title="Language Detection App")
st.title("🌍 Language Detection using NLP")
st.write("Enter a sentence below, and the model will predict the language.")

text_input = st.text_area("📝 Input Text:", height=150)

if st.button("Detect Language"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            transformed_input = vectorizer.transform([text_input])
            prediction = model.predict(transformed_input)
            st.success(f"🔍 Detected Language: **{prediction[0]}**")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
