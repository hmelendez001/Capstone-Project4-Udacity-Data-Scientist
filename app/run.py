import json
import plotly
import re
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
#import sklearn.external.joblib as extjoblib
import joblib
from sqlalchemy import create_engine
import sys
sys.path.append('../models')
from UrgencyWordExtractor import UrgencyWordExtractor
from MyTokenize import MyTokenize


app = Flask(__name__)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    This is the index webpage which displays visuals by way of Flask and the jinja2 engine to receive user input text and run against the model
    Parameters
    ----------
    None
    Returns
    -------
    string
        Return the generated HTML for this page
    """    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create a percentage count of all categories that are represented in the data
    category_counts = []
    category_names = []
    total_count = 0
    for column in df.columns:
        if (column not in ['id', 'message', 'original', 'genre']):
            category_count = df.loc[df[column] != 0, column].sum()
            if (category_count > 0):
                category_counts.append(category_count)
                category_names.append(column.replace("_", " "))
                total_count += category_count
    category_counts = [acount / total_count for acount in category_counts]
    
    # create visuals
    figures = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
            },
        },
        {
            'data': [
                Pie(
                    labels=category_names,
                    values=category_counts,
                ),
            ],

            'layout': {
                'title': 'Distribution of Represented Categories',
            },
        },
    ]
    
    # encode plotly figures in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly figures
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    This is the webpage that handles user input query and displays model results
    Parameters
    ----------
    None
    Returns
    -------
    string
        Return the generated HTML for this page
    """    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    The main function of this web application dashboard script
    Parameters
    ----------
    None
    Returns
    -------
    None
    """    
    app.run(host='0.0.0.0', port=3001, debug=True)
    #app.run()


if __name__ == '__main__':
    main()
