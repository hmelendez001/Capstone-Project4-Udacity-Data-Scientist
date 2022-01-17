import json
import plotly
import re
import numpy as np
import pandas as pd
import logging

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Line
import joblib
from sqlalchemy import create_engine
import sys


app = Flask(__name__)


# load data
engine = create_engine('sqlite:///../data/DatabaseCache.db')
#df = pd.read_sql_table('Messages', engine)
df = pd.read_csv('../data/initial_stocks.csv')
universe = pd.read_csv('../data/all_stocks_swingtradebot_dot_com.csv')
universe_list = universe['symbol'] + " | " + universe['name']
myuniverse = universe[universe['symbol'].isin(df['Symbol'].unique())]

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
    # create visuals
    symbols = df['Symbol'].unique()
    figures = [
        {
            'data': [
                Line(
                    x=df[df['Symbol'] == symbol]['Date'],
                    y=df[df['Symbol'] == symbol]['Adj Close'],
                    name=symbol
                ) for symbol in symbols
            ],

            'layout': {
                'title': 'Historical Prices',
                'yaxis': {
                    'title': "Price"
                },
                'xaxis': {
                    'title': "Date"
                },
            },
        },
    ]
    
    # encode plotly figures in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    ##app.logger.debug("******************** Symbols: {}".format(symbols.tolist()))
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    symbols = myuniverse[myuniverse['symbol'].isin(symbols)]
    symbols = universe_list##symbols['symbol'] + " | " + symbols['name']
    
    # render web page with plotly figures
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, symbols=json.dumps(symbols.tolist()), tables=[myuniverse.to_html(classes='table table-striped')], titles=myuniverse.columns.values)


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
