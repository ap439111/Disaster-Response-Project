"""
Clean the data and save it in SQLite database
Project: Disaster Response Pipeline 

Execute:
    > python process_data.py <path_to_messages_csv_file> <path_to_categories_csv_file> <path_to_sqllite_database>
    e.g. > python process_data.py messages.csv categories.csv disaster_response.db

Args:
    1) Path to load csv file containing messages (e.g. messages.csv)
    2) Path to load  csv file containing categories (e.g. categories.csv)
    3) Path to save SQLite database (e.g. disaster_response.db)
    
"""

import sys
import pandas as pd
import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to Load Messages Data and Categories data
    
    Args:
        messages_filepath: Path to the CSV file containing messages
        categories_filepath: Path to the CSV file containing categories
        
    Returns:
        df: Combined data containing messages and categories
    """
   
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories= pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Function to clean data. 
    
    Args:
        dataframe (df) that contains botht he message and category data
        
    Returns:
        dataframe (df) with 36 individual category columns with numeric values
        
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[[1]]
    category_colnames = list(category.split('-')[0] for category in row.values[0])
    categories.columns = category_colnames
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    #For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], join='inner', axis=1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Fucntion to Save Data to SQLite Database
    
    Args:
        df: Cleaned database containing messages and 36 categories 
        database_filename: Path to SQLite database file
        
    Returns:
            None
    """
    engine = create_engine('sqlite:///' + database_filename)
    table_name= os.path.basename(database_filename).replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False)
    


def main():
    """
     Main function to start the ETL pipeline. This function:
     
        1) Load Messages Data and Categories data from CSV files
        2) Clean Categories Data
        3) Save Data to SQLite Database
        
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()