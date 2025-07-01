import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


cv = CountVectorizer()

#data preprocessing
def preprocessing(df):
    """
    This function takes df along with the column name as an input and provided
    the corpus after preprocessing.
    """
    corpus = []
    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df[i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


#implementing the countvectorizer
def vectorizer(corpus, df):
    """
    This function takes three inputs and provided the final x and y values after applying the 
    vectorizer.
    df takes the dataframe along with the specified column name.
    """
    global cv
    x = cv.fit_transform(corpus).toarray()
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
def clean_input(input):
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
    x = cv.transform(corpus).toarray()
    return x

