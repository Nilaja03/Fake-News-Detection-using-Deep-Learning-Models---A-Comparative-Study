import streamlit as st
import pickle
import re

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load trained models
# -----------------------------
lstm_model = load_model("lstm_model.h5")
gru_model = load_model("gru_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Must match training
MAX_LEN = 300

# -----------------------------
# Text preprocessing (same as training)
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# -----------------------------
# Explainability: Important words
# -----------------------------
def get_important_words(text, model, tokenizer, max_len, top_n=5):
    words = text.split()

    # Original prediction
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    original_pred = model.predict(pad)[0][0]

    word_importance = []

    for word in set(words):
        modified_text = " ".join([w for w in words if w != word])
        seq_mod = tokenizer.texts_to_sequences([modified_text])
        pad_mod = pad_sequences(seq_mod, maxlen=max_len)
        new_pred = model.predict(pad_mod)[0][0]

        impact = abs(original_pred - new_pred)
        word_importance.append((word, impact))

    word_importance.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in word_importance[:top_n]]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection")
st.write("Enter a news article below to check whether it is **Real** or **Fake**, and see **why**.")

news_text = st.text_area("News Text", height=250)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess
        cleaned = clean_text(news_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Predictions
        lstm_pred = lstm_model.predict(padded)[0][0]
        gru_pred = gru_model.predict(padded)[0][0]

        lstm_label = "Real 🟢" if lstm_pred > 0.5 else "Fake 🔴"
        gru_label = "Real 🟢" if gru_pred > 0.5 else "Fake 🔴"

        # Confidence
        st.subheader("🔍 Prediction Results")
        st.write(f"**LSTM Prediction:** {lstm_label} (Confidence: {lstm_pred:.2f})")
        st.write(f"**GRU Prediction:** {gru_label} (Confidence: {gru_pred:.2f})")

        # Explanation
        st.subheader("🧠 Why this prediction?")

        lstm_words = get_important_words(cleaned, lstm_model, tokenizer, MAX_LEN)
        gru_words = get_important_words(cleaned, gru_model, tokenizer, MAX_LEN)

        st.write("**Important words (LSTM):**")
        st.write(lstm_words)

        st.write("**Important words (GRU):**")
        st.write(gru_words)

        # Human-readable explanation
        if lstm_pred <= 0.5:
            st.info(
                "The model classified this news as **Fake** due to the presence of "
                "sensational or misleading language patterns commonly found in fake news."
            )
        else:
            st.info(
                "The model classified this news as **Real** because the writing style and "
                "language resemble verified news articles."
            )