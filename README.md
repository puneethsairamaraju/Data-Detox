# Data Detox: Filtering the AI Fallout with Sophisticated Content Detection

> Detecting AI-Generated Text Using Ensemble Machine Learning & Transformer Models

---

## 📌 Overview

As generative AI models become increasingly sophisticated, distinguishing between **human-written** and **AI-generated** text has become a major challenge.

This project presents an advanced AI text detection system built using:

- Support Vector Machine (SVM)
- BERT (Bidirectional Encoder Representations from Transformers)
- DeBERTa (Decoding-enhanced BERT with Disentangled Attention)
- Flask-based Web Interface

Our goal is to provide **high-accuracy AI text detection** while minimizing false positives — especially in high-stakes environments like academia.

---

## 🚀 Motivation

Existing AI detection systems often struggle with:

- High false positive rates  
- Bias toward certain writing styles  
- Reduced reliability on longer essays  
- Risk of model collapse due to AI-generated training data  

This project addresses these issues using:

- Human-curated datasets  
- Balanced AI + human training data  
- Advanced transformer models  
- Ensemble modeling strategy  

---

## 🏗️ Architecture

### 1️⃣ Support Vector Machine (SVM)

- TF-IDF vectorization  
- Trained on 10,000+ essays  
- Accuracy: **98%**  
- Strong classical ML baseline  

---

### 2️⃣ BERT

- Fine-tuned for sequence classification  
- Trained on 46,000+ essays  
- Accuracy: **95%**  
- Excellent contextual understanding  

---

### 3️⃣ DeBERTa (Best Performing Model)

- Disentangled attention mechanism  
- Optimized using SGDClassifier  
- Trained on 44,000+ essays  
- Accuracy: **99%**  
- ROC-AUC: **1.00**  
- Lowest false positive rate  

---

## 📊 Dataset

### Human Essays
- Sourced from the Persuade Corpus  
- 500+ words per essay  
- Carefully curated and cleaned  

### AI-Generated Essays
Generated using:
- GPT models  
- LLaMA  
- Claude  
- Mistral  

### Dataset Properties

- Balanced (50% Human / 50% AI)  
- 300–500+ words per essay  
- Multi-LLM generation to reduce bias  
- 46,000+ rows total  

---

## 🖥️ Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| NLP | HuggingFace Transformers |
| ML | Scikit-learn |
| Deployment | Flask |
| Dev Environment | Google Colab Pro |
| Secure Exposure | Ngrok |
| UI | HTML, CSS |

---

## ⚙️ Implementation Workflow

1. Data Cleaning (citation removal, normalization)  
2. Tokenization (BERT / SentencePiece)  
3. TF-IDF Vectorization (for SVM)  
4. Model Training  
5. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  
6. Model Saving (joblib)  
7. Flask Web Deployment  

---

## 📈 Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|----------|
| SVM | 98% | 99% | 98% | 0.98 |
| BERT | 95% | 100% | 90% | 0.92 |
| DeBERTa | **99%** | 99% | 98% | **1.00** |

🏆 **Best Model: DeBERTa**

---

## 🌐 Web Application

Built using Flask.

### Features

- Text input form  
- Real-time prediction  
- Confidence score display  
- 4 classification categories:
  - Human  
  - Mixed (Human Dominant)  
  - Mixed (AI Dominant)  
  - AI Content  

---


cd data-detox-ai-detector
pip install -r requirements.txt
