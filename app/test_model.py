import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    ''' Given it is a tranformer we can return the self '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ''' apply starting_verb function to all values in X '''

        X_tagged = pd.Series(X).apply(self.starting_verb)

        return pd.DataFrame(X_tagged)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('disaster_response_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")
# Load from file
#with open("../models/classifier.pkl", 'rb') as file:
#    model = pickle.load(file)

#query=["How is it possible in my zone there is no acces card available to reveice food?"]
#query=["How is it possible we don't have access to food?"]
query=["the flood can be followed by an earthquake"]
# use model to predict classification for query
classification_labels = model.predict(query)
#classification_labels = model.predict([query])[0]
classification_results = dict(zip(df.columns[4:], classification_labels))
