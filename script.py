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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


#data preprocessing
vocab_size =14000
max_len = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

#defining a function to randomly take out a sample from HateSpeechDataset.csv
def sampling(df, n):
    random_sample = df.groupby("Label").sample(n, random_state = 0)
    random_sample.to_csv("Datasets/random_sample.csv", index = False)


def preprocessing(ser):
    """
    This function takes df along with the column name as an input and provided
    the corpus after preprocessing.
    """
    lst = ser.tolist()

    # Fit the tokenizer and set as global
    global tokenizer
    tokenizer.fit_on_texts(lst)

    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(lst)

    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    return padded_sequences

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
    This function takes a single input string and preprocesses it using the same
    tokenizer and padding as the main preprocessing function, so it can be used for prediction.
    """
    if 'tokenizer' not in globals():
        raise ValueError("Tokenizer has not been fit yet. Please fit the tokenizer on your dataset first.")
    # Tokenize and pad the input
    seq = tokenizer.texts_to_sequences([input])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded


