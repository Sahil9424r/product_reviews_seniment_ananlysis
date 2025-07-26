# 🛍️ Product Reviews Sentiment Analysis

This project focuses on analyzing customer reviews from **Flipkart** to classify them into three sentiment categories: **Positive**, **Negative**, and **Neutral**. The goal is to use machine learning models to understand public opinion on products based on their textual reviews.

## 📌 Project Overview

- **Problem**: Automatically detect the sentiment of customer reviews.
- **Approach**: Preprocess the reviews, apply word embeddings using Word2Vec, and train ML models.
- **Classes**: `Positive`, `Negative`, `Neutral`

## 📂 Dataset

- Dataset Source: [Flipkart Product Customer Reviews - Kaggle](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)
- File Used: `flipkart_customer_reviews.csv`
- Each record contains:  
  - `product_name`
  - `review_text`
  - `review_rating`
  - `review_sentiment` (Positive / Negative / Neutral)

## 🧹 Data Preprocessing

- Removed special characters, punctuation, numbers
- Converted text to lowercase
- Removed stopwords
- Tokenized and stemmed
- Applied **Word2Vec** to convert text into word embeddings

## 🔠 Word Embedding

- Used **Word2Vec** model (word2vec-google-news-300) to embed the reviews into numerical vectors of 300 dimenstion for model training.


## 🤖 Model Training

### ML Algorithms Used:
- **Logistic Regression**
- **Random Forest**
- **Descision Tree**

> All models were trained using the **Word2Vec embedded vectors**.
> Finally Random Forest model is selected

## 📈 Evaluation

- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- Tested sample predictions on real data


## Home Page
![Home Page](Sentimentpic/Screenshot (375).png)
## 😊 Positive Review
**Input**
![Positive Input](Sentimentpic/Screenshot (376).png)

**Predicted Output**
![Positive Output](Sentimentpic/Screenshot (377).png)

## 🤨 Negative Review
**Input**
![Negative Input](Sentimentpic/Screenshot (379).png)

**Predicted Output**
![Negative Output](Sentimentpic/Screenshot (380).png)

## 🤨 Neutral Review
**Input**
![Neutral Input](Sentimentpic/Screenshot (382).png)

**Predicted Output**
![Neutral Output](Sentimentpic/Screenshot (383).png)

