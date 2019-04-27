"""
PREPROCESSING DATA
Disaster Response Pipeline Project
Udacity - Data Science Nanodegree

Sample Script Execution:
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
Arguments:
    CSV file containing messages (e.g. disaster_messages.csv)
    CSV file containing categories (e.g. disaster_categories.csv)
    SQLite db name with file destination (e.g. DisasterResponse.db)
"""

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from filepath
    
    Arguments:
        messages_filepath: path to messages csv file
        categories_filepath: path to categories csv file
    Output:
        df: Loaded data as pd.DataFrame
    """
    # Read message data and categories data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge data sets
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    """
    Cleans Data from input df
    
    Arguments:
        df: raw data pd.DataFrame
    Outputs:
        df: clean pd.DataFrame
    """
    categories = df.categories.str.split(pat=';',expand=True)
    # pluck out first row which are the category names
    cat_names = categories.iloc[0,:]
    # get category names to use as column names
    category_colnames = cat_names.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    # convert categories
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # string -> integers
        categories[column] = categories[column].astype(np.int)
    # drop the original cat
    df = df.drop('categories',axis=1)
    # create the new df by concat the df and newly created categories
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    Saves the data into SQLite db
    
    Arguments:
        df: clean pd.DataFrame to be stored
        database_filename: destination for db
    Output:
        SQLite db
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False)
 

def main():
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