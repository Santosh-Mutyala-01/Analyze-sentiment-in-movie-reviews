# 🎬 Rotten Tomatoes Movie Reviews Sentiment Analysis

## 📌 Project Information  
- **Project Title:** Analyze sentiment in movie reviews  

---

## 🎯 Aim
- **Apply K-Means clustering** to group customer/movie review patterns.  
- **Build a sentiment analysis model** using Natural Language Processing (NLP).  

---

## 📝 Description
This project uses **Rotten Tomatoes critic reviews dataset** ([Kaggle link](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)) containing thousands of movie reviews.  

We apply:
1. **Text preprocessing** (cleaning, tokenization, stopword removal).  
2. **Feature engineering** using **TF-IDF**.  
3. **Supervised model**: Logistic Regression to predict **positive/negative sentiment**.  
4. **Baseline comparison** with VADER sentiment analyzer.  
5. **Unsupervised learning**: K-Means clustering to segment reviews.  

---

## 🛠️ Technologies Used
- **Python** (Pandas, NumPy)  
- **NLP Libraries**: NLTK, Scikit-learn, TF-IDF  
- **Visualization**: Matplotlib, Seaborn  
- **Clustering**: K-Means  

---

## 📂 Dataset
- Source: Kaggle – Rotten Tomatoes Movies and Critic Reviews  
- Total Records: ~17,000+ reviews  
- Columns of interest:
  - **Review Text** (critic review snippet)  
  - **Sentiment** (mapped to positive / negative)  

---

## 🔄 Project Workflow
1. **Load Dataset** → Inspect and select text + label columns.  
2. **Data Cleaning** → Remove HTML, punctuation, stopwords, lowercase conversion.  
3. **Exploratory Data Analysis (EDA)** → Label distribution, word frequencies.  
4. **Feature Engineering** → TF-IDF vectorization.  
5. **Model Training** → Logistic Regression.  
6. **Model Evaluation** → Accuracy, Precision, Recall, F1-score.  
7. **K-Means Clustering** → Group reviews based on textual similarity.  
8. **Predictions** → Test the model with custom user input.  

---

## 📊 Results

### Model Performance (Logistic Regression)
- **Accuracy:** 0.86  
- **Precision:** 0.84  
- **Recall:** 0.85  
- **F1-score:** 0.845  

### Baseline (VADER Sentiment Analyzer)
- Accuracy: ~0.72  

### K-Means Clustering (k=3)
- Cluster 0 → Mostly **Positive reviews**  
- Cluster 1 → Mixed / Neutral-like reviews  
- Cluster 2 → Mostly **Negative reviews**  

---

## 🚀 Example Predictions
```text
Input: "The movie was absolutely fantastic — great acting and direction!"
Output: Positive ✅

Input: "I hated this film. Boring and slow."
Output: Negative ❌
