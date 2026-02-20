# 📰 Neural Fake News Detector

An end-to-end Natural Language Processing (NLP) framework designed to classify news articles as **Real** or **Fake** using Deep Learning. This project features a comparative study between **LSTM** and **GRU** architectures, deployed as an interactive web application.

---

## 🚀 Key Features
* **Deep Learning Architectures:** Implemented and compared Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models.
* **Robust NLP Pipeline:** Includes custom text preprocessing (tokenization, padding, and stop-word removal) and optimized Embedding layers.
* **Real-time Inference:** A user-friendly interface for instantaneous news veracity checking.
* **Comparative Analytics:** Visual representation of model confidence and performance metrics.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow/Keras (or PyTorch), Scikit-learn, Pandas, NumPy
* **Preprocessing:** NLTK / SpaCy
* **Deployment:** Streamlit
* **Environment:** Jupyter Notebook / Google Colab

## 📊 Dataset
The models were trained on a merged dataset of over **40,000+** news articles, balanced for "True" and "Fake" labels. 
* **Input Features:** Full article text content.
* **Output:** Binary Classification (1: Real, 0: Fake).

## 🏗️ Architecture
<img width="794" height="433" alt="image" src="https://github.com/user-attachments/assets/5d2e2304-8c7a-4bf4-bf98-0ec44623fa75" />

1.  **Data Acquisition:** Cleaning and merging raw textual data.
2.  **Feature Engineering:** Converting text to sequences and applying padding for uniform input shape.
3.  **Model Training:**
    * **Embedding Layer:** Learned dense representations of words.
    * **Recurrent Layers:** LSTM/GRU for capturing long-term dependencies in text.
    * **Dense Layer:** Softmax/Sigmoid activation for final classification.
5.  **Deployment:** Model serialized and integrated into a Streamlit dashboard.

Results (app):

<img width="722" height="646" alt="image" src="https://github.com/user-attachments/assets/d3511a64-b25a-4042-90df-23c6067e4d5b" />

<img width="684" height="604" alt="image" src="https://github.com/user-attachments/assets/7fe212f3-e30a-41c8-bbd2-0b8a44689583" />

## 🚦 Getting Started

### Prerequisites
```bash
pip install tensorflow streamlit pandas scikit-learn
