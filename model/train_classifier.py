import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    '''
    Loads data from database as pd.DataFrame
    Input:
        database_filepath: File path to sql db
    Output:
        x: Message data 
        Y: Categories 
        category_names: 36 categories 
    '''
    # load data 
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)
    # Get features (messages) and categories (targets) 
    x = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return x, Y, category_names


def tokenize(text):
    '''
    Tokenizes and cleans text
    Input:
        text: message in text
    Output:
        tokenized: Tokenized, cleaned, and lemmatized text (ready for ML model)
    '''
    # Normalize Text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # Tokenize
    words = word_tokenize(text)
    # Remove Stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokenized = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    tokenized = [lemmatizer.lemmatize(w, pos='v').strip() for w in tokenized]
    
    return tokenized


def build_model():
    '''
    Builds an ML pipeline with ifidf, random forest, and gridsearch
    Random forest was chosen because of its speed and relatively accuracy.
    
    Input: 
        None
    Output:
        Results of GridSearchCV
    '''
    # This was done in a DataCamp NLP Course
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    # Use best parameters for model
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model performance using test data
    
    Input: 
        model: Model to be evaluated
        X_test: Test data (x)
        Y_test: Target data (Y)
        category_names: Labels for 36 categories
    Output:
        Results of classification and score
    '''
    Y_pred = model.predict(X_test)
    
    # Present the results and scores for the messages
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Save model in a pickle file 
    
    Input: 
        model: Model you wish to save
        model_filepath: output filepath
    Output:
        Pickle file with saved model
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, "wb"))


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