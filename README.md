# Supervised Machine Learning Pipeline   

## 📌 Project Overview

In this project, we build a supervised learning pipeline to classify political tweets based on their author. The dataset consists of labeled tweets from political figures and organizations, and the goal is to predict whether a tweet belongs to a conservative (e.g., `realDonaldTrump`, `GOP`) or liberal (e.g., `JoeBiden`, `TheDemocrats`) source.

The classification task is binary, and we apply several standard machine learning techniques including preprocessing, feature extraction using TF-IDF, and classification with Support Vector Machines (SVMs).

---

## ⚙️ Technologies & Libraries Used

- **Python 3.8+**
- **pandas, numpy** — Data manipulation and numerical processing  
- **NLTK** — Tokenization, POS tagging, lemmatization, and stopword handling  
- **scikit-learn**
  - `TfidfVectorizer` for feature extraction
  - `SVC` for Support Vector Classification
  - `accuracy_score` for evaluation
- **Regular expressions (re)** — Used for text normalization and cleaning

---

## 🧠 Main Components

### 1. Text Preprocessing
- Tweets are lowercased
- URLs and punctuation are removed
- Tokenization is applied
- POS tagging + lemmatization is used to normalize words

### 2. Feature Engineering
- A custom pipeline creates a **TF-IDF matrix** using processed tokens
- High-frequency stop words are removed
- Words that occur in fewer than 2 documents are filtered

### 3. Label Generation
- Tweets are labeled as:
  - `0` for conservative sources: `realDonaldTrump`, `mike_pence`, `GOP`
  - `1` for liberal sources: all others

### 4. Classifiers Implemented
- **MajorityLabelClassifier**: Baseline classifier that always predicts the most frequent class in training
- **Support Vector Machine (SVM)**: Used with various kernels (`linear`, `poly`, `rbf`, `sigmoid`) to learn from tweet features

### 5. Evaluation & Testing
- Models are evaluated on a held-out validation set using **accuracy**
- Best-performing kernel is selected
- The trained model is then used to classify tweets in the test dataset

---

## 🧪 How to Run

1. Install required packages:
    ```bash
    pip install pandas numpy scikit-learn nltk
    ```

2. Download NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')
    ```

3. Run the notebook:
    - Open `main.ipynb` in Jupyter
    - Follow the step-by-step workflow for training, evaluation, and predictions

4. For testing:
    - Test in `main.py` (contains all required function definitions) and `main.ipynb` (contains visualizations and responses)

---

## 📁 Files

- `main.ipynb` — Main notebook with code, analysis, and plots  
- `main.py` — Contains all function implementations for automated grading  
- `tweets_train.csv` — Training dataset with labeled tweets  
- `tweets_test.csv` — Unlabeled tweets to classify  
- `README.md` — Project documentation

---

## ✍️ Notes

- MajorityLabelClassifier is used as a simple baseline; SVM performance is compared against it
- The TF-IDF vectorizer used on training data is reused for test data transformation to maintain consistency