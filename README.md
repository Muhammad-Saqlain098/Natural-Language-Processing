#  Natural Language Processing (NLP) – Core Techniques in Python

This repository demonstrates **fundamental Natural Language Processing (NLP) techniques** implemented in Python using popular libraries such as **NLTK, Scikit-learn, and Gensim**.
It covers **text preprocessing, Bag of Words, TF-IDF, and Word Embeddings (Word2Vec)** with clear, beginner-friendly examples.

---

##  Project Objectives

* Understand **how raw text is cleaned and prepared** for NLP tasks
* Learn **vectorization techniques** used in Machine Learning models
* Implement **statistical and semantic text representations**
* Build a **strong NLP foundation** for sentiment analysis, classification, and deep learning

---

##  Technologies & Libraries Used

* **Python 3**
* **NLTK** – Text preprocessing & tokenization
* **Scikit-learn** – Bag of Words & TF-IDF
* **Gensim** – Word Embeddings (Word2Vec)
* **NumPy**

---

##  1. Text Preprocessing

Text preprocessing converts raw text into a clean, machine-readable format.

### Steps Implemented:

* Lowercasing
* Punctuation & number removal
* Tokenization
* Stopword removal
* Stemming
* Lemmatization
* Part-of-Speech (POS) tagging

 **Example Output:**

* Raw Text → Tokens → Cleaned Tokens → Stemmed/Lemmatized Words

 Libraries Used:

```python
nltk, string
```

---

##  2. Bag of Words (BoW)

Bag of Words represents text as a **frequency matrix of words**, ignoring grammar and word order.

### Features:

* Automatic tokenization
* Vocabulary generation
* Frequency-based vector representation

 **Tool Used:**

```python
CountVectorizer (scikit-learn)
```

 **Output:**

* Vocabulary list
* Bag-of-Words matrix

---

##  3. TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF improves BoW by reducing the importance of **common words** and increasing the weight of **important words**.

### Why TF-IDF?

* Handles common words better than BoW
* Improves ML model performance

 **Tool Used:**

```python
TfidfVectorizer (scikit-learn)
```

 **Output:**

* TF-IDF weighted matrix
* Normalized word importance scores

---

##  4. Word Embeddings – Word2Vec

Word2Vec represents words in **dense vector space**, capturing **semantic meaning**.

### Implemented Using:

* **Gensim**
* **Skip-Gram architecture**

### Features:

* Word vector generation
* Semantic similarity between words
* Context-based learning

 **Key Operations:**

* Train Word2Vec model
* Retrieve word vectors
* Compute word similarity

 **Example:**

```python
Similarity(nlp, language)
Similarity(nlp, class)
```

---

##  Project Structure

```
├── NLP (text_preprocessing).ipynb
├── NLP (BOW+TF-IDF).ipynb
├── NLP (Word Embeding).ipynb
├── README.md
```

---

## Use Cases

* Sentiment Analysis
* Text Classification
* Chatbots
* Information Retrieval
* Machine Learning & Deep Learning NLP pipelines

---

## Learning Outcomes

✔ Understand NLP preprocessing pipeline
✔ Apply BoW and TF-IDF correctly
✔ Learn semantic word representation using Word2Vec
✔ Build a solid foundation for advanced NLP tasks

---

## Author

**Abdullah Nazir**
AI | Machine Learning | NLP
Python • Scikit-Learn • NLTK • Gensim

---

## ⭐ If you like this project

Give it a **star ⭐** and feel free to fork or contribute!
