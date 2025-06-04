import streamlit as st
import pickle
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ----------------- Load Models ------------------ #
with open('Logisticmodel.pkl', 'rb') as f:
    category_model = pickle.load(f)

with open('model.pkl', 'rb') as f:
    fake_model = pickle.load(f)

# ----------------- Label Maps ------------------ #
category_labels = ['business', 'entertainment', 'politics', 'sport', 'tech']
fake_labels = {0: '🟥 FAKE', 1: '🟩 REAL'}

# ----------------- Utility Functions ------------------ #
def preprocess(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text.lower()

def show_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# ----------------- Page Styling ------------------ #
st.set_page_config(page_title="📰 Falsify", layout="wide", page_icon="🧠")
st.markdown("""
    <style>
        body { background-color: #f2f6fa; }
        .main { background-color: #f2f6fa; }
        .stTextArea textarea {
            font-family: monospace;
            font-size: 16px;
        }
        .stButton button {
            background-color: #0077b6;
            color: white;
            padding: 0.5em 1.5em;
            border-radius: 8px;
        }
        .stRadio > div { flex-direction: row; gap: 30px; }
        .box {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .result-title {
            font-size: 24px;
            font-weight: bold;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- App Title ------------------ #
st.markdown("<h1 style='text-align: center; color: #023047;'>🧠 Falsify</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Detect fake news and classify it into categories like business, sports, tech, etc.</h4>", unsafe_allow_html=True)
st.markdown("---")

# ----------------- Input ------------------ #
input_method = st.radio("Select Input Method", ['Paste News Article ✍️', 'Upload Text File 📄'])

text_input = ""
if input_method == 'Paste News Article ✍️':
    text_input = st.text_area("Enter your news article below:", height=300)
else:
    file = st.file_uploader("Upload a .txt file", type='txt')
    if file:
        text_input = file.read().decode("utf-8")

# ----------------- Analyze ------------------ #
if st.button("🚀 Analyze Article"):
    if text_input.strip() == "":
        st.warning("Please enter or upload a news article.")
    else:
        with st.spinner("Analyzing article... ✨"):
            cleaned_text = preprocess(text_input)

            # Category Prediction (using pipeline)
            category_pred = category_model.predict([cleaned_text])[0]
            category_name = category_labels[category_pred]

            # Fake/Real Prediction
            fake_pred = fake_model.predict([cleaned_text])[0]
            fake_result = fake_labels[fake_pred]
            fake_color = "green" if fake_pred == 1 else "red"

        # ------------- Display Results ------------- #
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='box'>", unsafe_allow_html=True)
            st.markdown("### 🧠 News Category Prediction")
            st.markdown(f"<p class='result-title' style='color: #0077b6;'>{category_name.title()}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='box'>", unsafe_allow_html=True)
            st.markdown("### 🚨 Fake News Detection")
            st.markdown(f"<p class='result-title' style='color: {fake_color};'>{fake_result}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ------------- Word Cloud ------------- #
        st.markdown("### ☁️ Word Cloud")
        st.info("Visual representation of word frequency in the article.")
        show_wordcloud(cleaned_text)


# ----------------- Footer ------------------ #
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit · Model: Logistic Regression + TF-IDF · Detection: Binary Classifier</p>", unsafe_allow_html=True)
