#%%
import numpy as np
import pandas as pd
import nltk
import sklearn 
import string
import re # helps you filter urls
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#%%
# Whether to test your Part 9 for not depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass for below functions

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    # Step 1: Convert to lower case
    text = text.lower()
    
    # Step 2: Remove URLs
    text = re.sub(r'http:/\S+|www\S+|https:/\S+', '', text, flags=re.MULTILINE)
    
    # Step 3: Remove or handle punctuation
    # Handle specific punctuation cases and remove others
    text = re.sub(r"'s", '', text)  # Remove 's
    text = re.sub(r"'", '', text)  # Replace other apostrophes with ''
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text) # Replace any punctuation with a space
    
    # Step 4: Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Step 5: Lemmatize tokens based on POS
    lemmatized_tokens = []
    pos_tags = nltk.pos_tag(tokens)  # Get POS tags
    
    for word, tag in pos_tags:
        # Get the first letter of the POS tag
        first_letter = tag[0]
        pos = posMapping.get(first_letter, 'n')  # Default to noun if not found
        try:
            lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
            lemmatized_tokens.append(lemmatized_word)
        except Exception:
            continue  # Ignore words that cannot be lemmatized

    return lemmatized_tokens
    
#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    # Apply the process function to the 'text' column
    df['text'] = df['text'].apply(lambda x: process(x, lemmatizer))
    return df
    
#%%
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        processed_tweets: pd.DataFrame: processed tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords (after processing)
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    # sort stop_words set first

    # Initialize the TfidfVectorizer with the required parameters
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        lowercase=False,  # Convert to lowercase
        tokenizer=lambda x: x,  # The input is already tokenized
        stop_words=list(sorted(stop_words)),  # Ignore the stop words
        min_df=2  # Filter out words that occur in less than 2 documents
    )
    
    # Fit the vectorizer to the processed tweets and transform the text into a sparse matrix
    feature_matrix = vectorizer.fit_transform(processed_tweets['text'])
    
    return vectorizer, feature_matrix

#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        processed_tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    # Define the screen names that should be labeled as 0
    label_0_names = {'realDonaldTrump', 'mike_pence', 'GOP'}
    
    # Create the labels by checking if each screen_name is in the label_0_names set
    labels = processed_tweets['screen_name'].apply(lambda name: 0 if name in label_0_names else 1)
    
    # Convert the result to a numpy array
    return labels.to_numpy(dtype=int)
#%%
class MajorityLabelClassifier():
    """
    A classifier that predicts the mode of training labels
    """
    def __init__(self):
        """
        Initialize your parameter here
        """
        self.mode_label = None  # To store the mode of labels
        
    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        i.e. store your learned parameter
        """
        # Count occurrences of each label in y
        label_counts = {}
        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # Find the label with the maximum count (the mode)
        self.mode_label = max(label_counts, key=label_counts.get)
    
    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
        # Create an array with the same length as X, filled with the mode label
        return np.full(X.shape[0], self.mode_label)

#%%
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.SVC: classifier learnt from data
    """  
    # Initialize SVC with specified kernel
    classifier = SVC(kernel=kernel)
    
    # Fit the classifier on the training data
    classifier.fit(X_train, y_train)
    
    return classifier

#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    # Use the classifier to predict labels on the validation set
    y_pred = classifier.predict(X_validation)
    
    # Calculate and return accuracy
    accuracy = accuracy_score(y_validation, y_pred)
    return accuracy

#%%
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.SVC: classifier learned
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    # Process the unlabeled tweets
    unlabeled_tweets_processed = process_all(unlabeled_tweets)

    # Use the tfidf vectorizer to transform the raw text of the unlabeled tweets
    X_unlabeled = tfidf.transform(unlabeled_tweets_processed['text'])
    
    # Use the trained classifier to predict class labels for the unlabeled tweets
    y_pred = classifier.predict(X_unlabeled)
    
    return y_pred