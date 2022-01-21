import json
import plotly
import re
import numpy as np
import pandas as pd
import logging
import datetime

from flask import Flask
from flask import render_template, request, session, jsonify
from flask_session import Session
from plotly.graph_objs import Line
import joblib
from sqlalchemy import create_engine
import sys


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


# load data
engine = create_engine('sqlite:///../data/DatabaseCache.db')
#df = pd.read_sql_table('Messages', engine)
df = pd.read_csv('../data/initial_stocks.csv')
universe = pd.read_csv('../data/all_stocks_swingtradebot_dot_com.csv', parse_dates=['begin_train', 'end_train'])
universe_list = universe['symbol'] + " | " + universe['name']
myuniverse = universe[universe['symbol'].isin(df['Symbol'].unique())]

@app.template_filter()
def format_datetime(value, format='%Y-%m-%d'):
    """
    Returns True if the given parameter can be converted into the given date format, False otherwise
    This function is used inside our html code to format date values into our desired date format, e.g.:
    <td>{{row[7] | format_datetime}}</td>
    Parameters
    ----------
    value (str) - the date string to format
    format (str) - optional string format
    Returns
    -------
    bool
        True if the given parameter can be converted into the given date format, False otherwise
    Comments
    --------
    Credit to StackOverflow users Tom Burrows and tux21b for this solutions used by jinja2 to format dates
    at https://stackoverflow.com/a/4830620/2788414 and to Michael Cho for this type of implementation as 
    written in: https://michaelcho.me/article/custom-jinja-template-filters-in-flask
    """
    if isinstance(value, datetime.date):
        return value.strftime(format)
    else:
        return datetime.datetime.strptime(value, "%m/%d/%Y").strftime(format)

def is_int(aString: str) -> bool:
    """
    Returns True if the given parameter can be converted into a numeric integer, False otherwise
    Parameters
    ----------
    None
    Returns
    -------
    bool
        True if the given parameter can be converted into a numeric integer, False otherwise
    """    
    try:
        int(aString)
        return True
    except ValueError:
        return False

def parseDateRange(aString: str):
    """
    Returns start and end dates for the given date range string
    Parameters
    ----------
    aString (str) - a string date range like '2021-01-01 - 2021-12-31'
    Returns
    -------
    start (np.datetime64) - starting date
    end (np.datetime64) - ending date
    """
    separator = ' - '
    x = aString.find(separator)
    startStr = aString[0:x].rstrip()
    endStr = aString[x+len(separator):].rstrip()

    ##start = np.datetime64(datetime.datetime.strptime(startStr, "%Y-%m-%d"))
    ##end = np.datetime64(datetime.datetime.strptime(endStr, "%Y-%m-%d"))
    start = np.datetime64(startStr, 'D')
    end = np.datetime64(endStr, 'D')

    return start, end

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
    Comments
    --------
    Credit to users Cody Gray and Ashwin Srinivas on StackOverflow for pointing out that it's better to store a Pandas DataFrame as a dictionary list
    in the Session state, however, I am doing dictionary keys since I need the key for removing items: at https://stackoverflow.com/a/63332386/2788414
    We would not use Session for a production system, instead we would cache state in some other way that is more secure
    More on why Flask Session is not secure: https://blog.miguelgrinberg.com/post/how-secure-is-the-flask-user-session
    """    
    # create visuals
    ##app.logger.debug("------------------dtypes: {}".format(myuniverse.dtypes))
    # handle session state to get "myuniverse" of stocks
    if not session.get("myuniverse"):
        session["myuniverse"] = myuniverse.to_dict()

    # handle removing a name
    if ('stockIdToDelete' in request.args.keys()):
        stockIdToDelete = request.args['stockIdToDelete']
        deltable = pd.DataFrame(session["myuniverse"])
        if len(stockIdToDelete) > 0 and is_int(stockIdToDelete) and int(stockIdToDelete) in deltable.index:
            ##app.logger.debug("------------------stockIdToDelete: {}".format(stockIdToDelete))
            deltable.drop(int(stockIdToDelete), inplace=True)
            session["myuniverse"] = deltable.to_dict()

    # handle adding a new name
    if ('stockText' in request.args.keys() and 'daterange' in request.args.keys()):
        stockToAdd = request.args['stockText']
        daterangeToAdd = request.args['daterange']
        if len(stockToAdd) > 0 and len(daterangeToAdd) > 0 and stockToAdd.find('|') > 1:
            x = stockToAdd.find('|')
            tickerToAdd = stockToAdd[0:x-1].rstrip()
            addtable = pd.DataFrame(session["myuniverse"])
            existing_df = addtable[addtable['symbol'] == tickerToAdd]
            if (existing_df.shape[0] == 0):
                ##app.logger.debug("******************** tickerToAdd: {}, daterangeToAdd: {}".format(tickerToAdd, daterangeToAdd))
                newStock = universe[universe['symbol'] == tickerToAdd]
                # TODO: Handle when a ticker is given not in our universe
                if (newStock.shape[0] > 0):
                    ##app.logger.debug("+++++++++++++++++++ tickerToAdd: {}, newStock: {}".format(tickerToAdd, newStock.tail()))
                    ##app.logger.debug("------------------addtable.tail(): {}".format(addtable.tail()))
                    start, end = parseDateRange(daterangeToAdd)
                    newStock.at[newStock.index.max(), 'begin_train'] = start
                    newStock.at[newStock.index.max(), 'end_train'] = end
                    ##app.logger.debug("------------------newStock.tail(): {}".format(newStock.tail()))
                    addtable = addtable.append(newStock).sort_values(by=['symbol'])
                    session["myuniverse"] = addtable.to_dict()
                    ##app.logger.debug("+++++++++++++++++++ myuniverse: {}, \n+++++++++++++++++inPlace {}".format(myuniverse.tail(), myuniverse.append(newStock, sort=True).tail()))

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

    symbols = universe_list

    mytable = pd.DataFrame(session["myuniverse"])
    ###app.logger.debug("******************** mytable1: {}".format(mytable.tail()))

    mytable['myindex'] = mytable.index
    html_table = mytable[['myindex', 'symbol', 'name', 'close_price', 'volume', 'fifty_two_week_low', 'fifty_two_week_high', 'begin_train', 'end_train']]

    ##app.logger.debug("******************** Symbols: {}".format(html_table.to_numpy().tolist()))
    
    # render web page with data
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, symbols=json.dumps(symbols.tolist()), tables=html_table.to_numpy().tolist())

# web page that handles allowing predictions on existing stock models
@app.route('/predict')
def predict():
    """
    This is the webpage that allows price predictions based on trained models
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
    #classification_labels = model.predict([query])[0]
    #classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'predict.html',
        query=query
        , tables=[myuniverse.to_html(classes='table table-striped table-hover')], titles=myuniverse.columns.values
    )

# web page that handles showing additional recommendations based on existing stocks
@app.route('/recommend')
def recommend():
    """
    This is the webpage that displays additional recommendations based on existing stocks
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
    #classification_labels = model.predict([query])[0]
    #classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'recommend.html',
        query=query
        , tables=[myuniverse.to_html(classes='table table-striped table-hover')], titles=myuniverse.columns.values
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
