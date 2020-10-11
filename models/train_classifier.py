"""
Script for Classifier Trainer
Project: Disaster Response Pipeline 

Execute:
    > python train_classifier.py <path_to_sqllite_db>  <path_to_save_pickle_file>
  
    e.g > python train_classifier.py ../data/disaster_response.db classifier.pkl

Args:
    1) Path to load SQLite database file (e.g. ../data/disaster_response.db)
    2) Path to save ML model parameters as pickle file (e.g. classifier.pkl)
    
"""
import sys
import os
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    """
    Function to load data from the SQLite database
    
    Args:
        database_filepath: Path to SQLite destination database (e.g. disaster_response.db)
        
    Returns:
            X: A dataframe containing Features
            Y: A dataframe containing Labels
            category_names: List of categories
            
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    table_name= os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    '''X,Y variable'''
    X = df['message']
    Y = df.iloc[:,4:]
    
    '''The related column has few values of 2. Our classifier has two Outputs: 0 and 1.
    So, replacing 2 values by the max value (which is 1)'''
    
    Y['related']= Y['related'].map(lambda x: 1 if x == 2 else x)
    
    category_names = Y.columns.values
    
    return X,Y,category_names


def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text: Text messages
    Output:
        clean_tokens: List of tokens for the text
    """
    
    '''Replace all urls with a urlplaceholder string'''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    '''Extract all the urls from the text '''
    urls = re.findall(url_regex, text)
    
    '''Replace url with a urlplaceholder '''
    for url in urls:
        text = text.replace(url, 'urlplaceholder')

    '''Extract the word tokens from the provided text'''
    tokens = nltk.word_tokenize(text)
    
    '''Lemmanitizer to remove inflectional and derivationally related forms of a word'''
    lemmatizer = nltk.WordNetLemmatizer()

    '''List of clean tokens'''
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]
    
    return clean_tokens
   

    '''Build a custom transformer which builds starting verb of the sentence'''

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

    
def build_model():
    
    """
    Funciton to build a pipeline with custom transformer as a feature
    
    Args:
        None
        
    Returns:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        Applies a Gridsearch to optimize the parameters
        
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    param_grid = {'classifier__estimator__learning_rate': [0.1, 0.01, 0.005],
                  'classifier__estimator__n_estimators': [10, 40, 50],
                  'features__text_pipeline__vect__max_features': (None, 5000, 10000),
                  'features__text_pipeline__tfidf__use_idf': (True, False)}
    
    cv = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1_micro', n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to predict from the ML pipleine and prints the model performance (precision, f1-score)
    
    Args:
        model: ML pipline 
        X_test: the test set of features
        Y_test: the test set of labels
        category_names: List of categories
        
    Returns:
        None
        
    """
    
    y_test_pred= model.predict(X_test)
    
    print(classification_report(Y_test, y_test_pred, target_names=category_names))
    



def save_model(model, model_filepath):
    """
    Function to save the ML pipeline as a pickle file 
    
    Args:
        model: ML pipleline
        model_filepath: path to save the .pkl file
    
    """
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
