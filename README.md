# 🐦 Twitter Sentiment Analysis

A Python-based text mining project that analyzes tweets and classifies them into **Positive**, **Negative**, **Neutral**, or **Irrelevant** categories.  
This project uses **Natural Language Processing (NLP)** techniques, **Bag of Words (BoW)**, **n-grams**, and **Logistic Regression** to perform accurate sentiment classification.

---

# 📑 Table of Content
- [Features](#features)
- [Technology Stack](#technology-stack)
- [How-It-Works](#how-it-works-)
- [Dataset](#dataset)

---

# ✨ Features

**Text Preprocessing** — Cleans and normalizes tweets using lowercase conversion, regex cleaning, and tokenization.  

**WordCloud Visualization** — Generates visual word clouds for each sentiment type to understand frequently used terms.  

**Sentiment Classification** — Classifies tweets into Positive, Negative, Neutral, and Irrelevant categories.  

**Bag of Words & n-grams** — Converts text into numerical vectors using BoW and analyzes word patterns up to 4-grams.  

**Performance Evaluation** — Evaluates models using accuracy metrics on training and validation datasets.  

**Scalable & Extendable** — You can integrate more ML models (like XGBoost or SVM) to improve performance.

---

# 🧠 Technology Stack

### Core Libraries:
- **Python 3** — Core programming language.  
- **Pandas** — For data loading and cleaning.  
- **NumPy** — For numerical computations.  
- **Matplotlib & Seaborn** — For data visualization.  
- **WordCloud** — For visualizing the most frequent words.  
- **NLTK** — For tokenization and stopword removal.  
- **Scikit-learn** — For feature extraction and model building.  
- **XGBoost** — Optional advanced model for classification.

---

# ⚙️ How It Works 🧩

The sentiment analysis workflow follows a **text mining and machine learning** pipeline:

1. **Data Loading** — Import the training and validation datasets.  
2. **Text Cleaning** — Convert text to lowercase, remove special characters, and tokenize words.  
3. **WordCloud Generation** — Visualize frequent terms in Positive, Negative, Neutral, and Irrelevant tweets.  
4. **Feature Extraction** — Convert text data into numerical form using **CountVectorizer (BoW)** and **n-grams**.  
5. **Model Training** — Train **Logistic Regression** models using the extracted features.  
6. **Evaluation** — Compute accuracy on both test and validation sets to assess performance.

---

# 📊 Dataset

The project uses the **Twitter Sentiment Dataset**, which contains labeled tweets categorized into four sentiment types:

- `twitter_training.csv` — Training dataset  
- `twitter_validation.csv` — Validation dataset  

Each record includes:
- **id** — Unique identifier for each tweet  
- **information** — Topic or brand related to the tweet  
- **type** — Sentiment label (Positive / Negative / Neutral / Irrelevant)  
- **text** — Tweet content  
