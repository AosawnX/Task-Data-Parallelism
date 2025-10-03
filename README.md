# 🚀 Data & Task Parallelism for Machine Learning Applications

This repository contains the implementation and report for **Assignment 2** of the *Parallel and Distributed Computing* course. The project explores the impact of **data parallelism (GPU/CUDA)** and **task parallelism (CPU multithreading)** on machine learning tasks.

---

## 📂 Project Structure
- **/fraud_detection/** – Preprocessing, EDA, and CUDA/CPU MLP implementation.
- **/imdb_sentiment/** – LSTM sentiment analysis with CUDA acceleration vs CPU task parallelism.
- **/twitter_sentiment/** – COVID-19 tweets, preprocessing + TF–IDF + Logistic Regression (CPU) vs LSTM (GPU).
- **/report/** – LaTeX report with Introduction, Methodology, EDA, Experimental Setup, Results, and Annexures.

---

## ⚙️ Tech Stack
- **Languages:** Python
- **Libraries:** PyTorch, scikit-learn, Pandas, NumPy, TensorFlow (preprocessing), Seaborn/Matplotlib, WordCloud
- **Parallelism Tools:** 
  - CUDA (GPU data parallelism)
  - Python threading / DataLoader workers (CPU task parallelism)

---

## 📊 Key Results
| Dataset              | CPU Runtime         | GPU Runtime | Speedup   | Accuracy/F1 |
|-----------------------|--------------------|-------------|-----------|-------------|
| Fraud Detection       | [CPU time]         | [GPU time]  | ~10–20×   | F1 ~0.76    |
| IMDB Sentiment        | 405s / 985s        | 8.79s       | ~100×     | Acc ~0.86   |
| Twitter Sentiment     | 121.09s            | 6.41s       | ~19×      | Acc ~0.82   |

---

## 📑 Report
The full report includes:
- Introduction & Problem Statement
- Objectives
- Methodology (with EDA visuals)
- Experimental Setup
- Evaluation Metrics
- Results & Observations
- Annexures (GitHub repo + references)

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
