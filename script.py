import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


#data preprocessing

tfv = TfidfVectorizer()
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

#defining a function to randomly take out a sample from HateSpeechDataset.csv
def sampling(df, n):
    random_sample = df.groupby("Label").sample(n, random_state = 0)
    random_sample.to_csv("Datasets/random_sample.csv", index = False)


def preprocessing(df):
    """
    This function takes df along with the column name as an input and provided
    the corpus after preprocessing.
    """
    corpus = []
    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df[i])
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in stop_words]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


#implementing the countvectorizer

def vectorizer(corpus, df):
    """
    This function takes the preprocessed corpus and the dataframe column (labels),
    applies the TfidfVectorizer, and returns the feature matrix x and target vector y.
    """
    global tfv
    x = tfv.fit_transform(corpus).toarray()
    y = df.values
    return x, y


#splitting the test and training data

def split(x, y):
    """
    This function takes the x and y vectors after preprocessing and 
    splits them into training and test sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test


#evaluating the model's performance
def evaluation(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    result = f"The accuracy score for the actual and predicted values is {accuracy} \nand the confusion matrix is {cm}"
    return result

#preprocessing single input
def clean_input(input, vectorizer):
    """
    This function takes a single output 
    and preprocesses it to be suitable for prediction.
    """
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', input)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    x = vectorizer.transform(corpus).toarray()
    return x

