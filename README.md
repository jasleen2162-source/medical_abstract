# 🩺 Medical Abstract Sentence Classifier using PubMed RCT

A Natural Language Processing (NLP) project that classifies each sentence in a PubMed Randomized Controlled Trial (RCT) abstract into its rhetorical role: **Background, Objective, Methods, Results, or Conclusions**. The project demonstrates how AI can help researchers and clinicians quickly understand medical literature.

> **Disclaimer:** This project is developed for educational purposes and is not intended to provide medical advice or clinical decision support.

---

## 📌 Features

- Sentence-level medical text classification
- TF-IDF + Logistic Regression baseline model
- Deep learning model (BiLSTM/Text-CNN/Transformer)
- Text preprocessing and tokenization
- Model evaluation using standard NLP metrics
- Confusion Matrix and Classification Report
- Demo for color-coded medical abstract prediction

---

## 🏥 Business Use Cases

- Faster medical literature review
- Biomedical document mining
- Medical research summarization
- Educational Healthcare NLP application

---

## 📂 Dataset

**Dataset:** PubMed 20k RCT (Recommended) / PubMed 200k RCT

Each sentence is labeled as one of:

- Background
- Objective
- Methods
- Results
- Conclusions

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📊 Evaluation Metrics

- Accuracy
- Macro F1-score
- Weighted F1-score
- Precision
- Recall
- Confusion Matrix
- Classification Report

