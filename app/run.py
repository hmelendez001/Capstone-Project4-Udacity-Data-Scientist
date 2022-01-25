import json
import plotly
import re
import numpy as np
import pandas as pd
import logging
import os.path
import datetime

from flask import Flask
from flask import render_template, request, session, jsonify
from flask_session import Session
from plotly.graph_objs import Line
import joblib
from sqlalchemy import create_engine
import sys

from StockDateValidator import StockDateValidator
from StockHistory import StockHistory
from StockPricePredictor import StockPricePredictor
from StockModelLinear import StockModelLinear


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# In order to minimize calls to the API we will cache or store calls per day per symbol, 
# meaning we will only call the API once per day per stock symbol
stock_hash = {}

# In order for get_stock_predictions to use the model trained by get_stock_histories we will have to cache these 
# trained models here per stock symbol. Since we need both the model itself plus the dates for which it was trained 
# we will also create a class that encapsulates these concepts into one object that we cache by symbol.
model_hash = {}

# load data, not using SQL Lite at this time in favor of just using Session state, instead initializing data from a file
# engine = create_engine('sqlite:///../data/DatabaseCache.db')
#df = pd.read_sql_table('Messages', engine)
# Relative paths do not work so great with Heroku
d = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(d, '..')
datapath = os.path.join(datapath, 'data')
df = pd.read_csv(os.path.join(datapath, 'initial_stocks.csv'), parse_dates=['Date'])

# For now the "known universe" of stocks are the listed US stocks from https://swingtradebot.com/ that we ared in from a file
# at this time. In the future we would call an API like Yahoo! Finance for look ahead search
universe = pd.read_csv(os.path.join(datapath, 'all_stocks_swingtradebot_dot_com.csv'), parse_dates=['begin_train', 'end_train'])
universe_list = universe['symbol'] + " | " + universe['name']

def initialize_stock_cache():
    '''
    INPUT:
    None 
    
    OUTPUT:
    symbols (list) - List of symbols in our initial data
    from_date (datetime.date) - Starting date of data
    to_date (datetime.date) - Ending date of data
    
    Description:
    This function initializes stock_hash from our initial data so we can call StockHistory.getStockHistories for the initial symbols
    '''
    # Get today's date because we need to use this for our keys
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    symbols = df['Symbol'].unique()

    for symbol in symbols:
        key = today + "|" + symbol
        stock_hash[key] = df[df['Symbol'] == symbol]

    return symbols, df.Date.min(), df.Date.max()

def CreateStockModel(symbol: str, df_symbol: pd.DataFrame, newFromDate: datetime.date, newToDate: datetime.date):
    '''
    INPUT:
    symbol (str) - The stock ticker to get historical data for
    df_symbol (Pandas.DataFrame) - The underlying Pandas DataFrame of train and test data
    newFromDate (datetime.date) - The starting date range to get data for 
    newToDate (datetime.date) - The ending date range to get data for 
    
    OUTPUT:
    model (StockModel) - The model of choice for our tests
    
    Description:
    This function determines which child class StockModel to use for our predictions
    '''
    #TODO: add API call to get symbol *not* in universe
    df_info = universe[universe['symbol'] == symbol]
    idx = df_info.index.min()
    df_info.at[idx, 'begin_train'] = newFromDate
    df_info.at[idx, 'end_train'] = newToDate
    return StockModelLinear(symbol, df_symbol, newFromDate, newToDate, df_info)

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
    elif isinstance(value, np.datetime64):
        ts = pd.to_datetime(str(value)) 
        return ts.strftime(format)
    elif isinstance(value, str):
        return datetime.datetime.strptime(value, "%m/%d/%Y").strftime(format)
    else:
        return str(value)

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
    # handle session state to get "stock_hash" (historical data) and "model_hash" (trained cached model data)
    if not session.get("stock_hash"):
        session["stock_hash"] = stock_hash
    if not session.get("model_hash"):
        session["model_hash"] = model_hash

    error_msg = ''
    # Very first time we need to initialize the stock_hash and model_hash
    MyStockHash = session["stock_hash"]
    if len(MyStockHash) == 0:
        symbols, from_date, to_date = initialize_stock_cache()
        hist = StockHistory(stock_hash, model_hash)
        error_msg, bool_ret, MyModelHash = hist.getStockHistories(symbols, from_date, to_date, CreateStockModel)
        MyStockHash = stock_hash
        session["stock_hash"] = stock_hash
        session["model_hash"] = MyModelHash

    MyModelHash = session["model_hash"]
    hist = StockHistory(MyStockHash, MyModelHash)
    # handle removing a name
    if ('stockSymbolToDelete' in request.args.keys()):
        stockSymbolToDelete = request.args['stockSymbolToDelete']
        if len(stockSymbolToDelete) > 0 and stockSymbolToDelete in MyModelHash.keys():
            ##app.logger.debug("******************** DELETE symbols: {}".format(stockSymbolToDelete))
            MyModelHash = hist.dropModelFromCache(stockSymbolToDelete)
            app.logger.debug("++++++++++++++ DELETE {} remaining Symbols: {}".format(stockSymbolToDelete, list(MyModelHash.keys())))
            session["model_hash"] = MyModelHash

    # handle adding a new name
    if ('stockText' in request.args.keys() and 'daterange' in request.args.keys()):
        stockToAdd = request.args['stockText']
        daterangeToAdd = request.args['daterange']
        if len(stockToAdd) > 0 and len(daterangeToAdd) > 0:
            if stockToAdd.find('|') > 1:
                x = stockToAdd.find('|')
                tickerToAdd = stockToAdd[0:x-1].rstrip()
            else:
                tickerToAdd = stockToAdd
            start, end = StockDateValidator.parseDateRange(daterangeToAdd)
            df_universe = universe[universe['symbol'] == tickerToAdd]
            app.logger.debug("++++++++++++++ ADDING Symbols: {}\n{}".format(tickerToAdd, df_universe.tail()))
            if df_universe.shape[0] == 0:
                error_msg = "Stock ticker '" + tickerToAdd + "' not a valid US listed ticker."
            else:
                error_msg, bool_ret, MyModelHash = hist.getStockHistories([tickerToAdd], start, end, CreateStockModel)
                if not bool_ret:
                    error_msg = "Stock " + tickerToAdd + ": " + error_msg
                    app.logger.error("++++++++++++++ getStockHistories: {}: {}".format(bool_ret, error_msg))
                ##app.logger.debug("++++++++++++++ ADDED Symbols: {}".format(list(MyModelHash.keys())))
                session["model_hash"] = MyModelHash

    symbols = list(MyModelHash.keys())
    figures = [
        {
            'data': [
                Line(
                    x=MyModelHash[symbol].df['Date'],
                    y=MyModelHash[symbol].df['Adj Close'],
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
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    mytable = pd.DataFrame()
    for symbol in symbols:
        mytable = mytable.append(MyModelHash[symbol].df_info)
    mytable.sort_values(by=['symbol'], inplace=True)

    mytable['myindex'] = mytable.index
    html_table = mytable[['myindex', 'symbol', 'name', 'close_price', 'volume', 'fifty_two_week_low', 'fifty_two_week_high', 'begin_train', 'end_train']]

    ##app.logger.debug("******************** Symbols: {}".format(html_table.to_numpy().tolist()))
    
    # render web page with data
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, symbols=json.dumps(universe_list.tolist()), tables=html_table.to_numpy().tolist(), error_msg = error_msg)

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
    MyModelHash = session["model_hash"]
    mytable = pd.DataFrame()
    for symbol in MyModelHash.keys():
        mytable = mytable.append(MyModelHash[symbol].df_info)
    mytable.sort_values(by=['symbol'], inplace=True)

    mytable['myindex'] = mytable.index
    html_table = mytable[['myindex', 'symbol', 'name', 'close_price', 'volume', 'fifty_two_week_low', 'fifty_two_week_high', 'eps', 'div_yield', 'sector', 'begin_train', 'end_train']]

    # handle prediction prices for a given name
    error_msg = ''
    ids = []
    figures = []
    evals = {}
    if ('stockSymbolToPredict' in request.args.keys() and 'dateRangeToPredict' in request.args.keys()):
        stockSymbolToPredict = request.args['stockSymbolToPredict']
        dateRangeToPredict = request.args['dateRangeToPredict']
        if len(stockSymbolToPredict) > 0 and stockSymbolToPredict in MyModelHash.keys():
            start, end = StockDateValidator.parseDateRange(dateRangeToPredict)
            predict = StockPricePredictor(MyModelHash)
            error_msg, df_ret = predict.getStockPredictions([stockSymbolToPredict], start, end)
            if (len(error_msg) > 0):
                app.logger.error("********************: symbol {} from {} to {}: {}".format(stockSymbolToPredict, start, end, error_msg))
            else:
                ##app.logger.debug("********************: symbol {} from {} to {}:\n{}".format(stockSymbolToPredict, start, end, df_ret.tail()))
                symbol = stockSymbolToPredict
                evals, actuals, predicted = MyModelHash[symbol].evaluateModel()
                graph_title = symbol + ": " + MyModelHash[symbol].df_info.at[MyModelHash[symbol].df_info.index.max(), 'name'] + ' Prices'
                figures = [
                    {
                        'data': [
                            Line(
                                x=MyModelHash[symbol].df['Date'],
                                y=MyModelHash[symbol].df['Adj Close'],
                                name="Historical " + symbol
                            ),
                            Line(
                                x=df_ret['Date'],
                                y=df_ret['Adj Close'],
                                name="Predicted " + symbol
                            ),
                            Line(
                                x=MyModelHash[symbol].df[MyModelHash[symbol].df['Date'] > MyModelHash[symbol].last_trained_date]['Date'],
                                y=predicted,
                                name="Tested " + symbol
                            )
                        ],

                        'layout': {
                            'title': graph_title,
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
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)   
                
    # render web page with data
    return render_template('predict.html', ids=ids, figuresJSON=figuresJSON, symbols=json.dumps(universe_list.tolist()), tables=html_table.to_numpy().tolist(), evals=evals, error_msg = error_msg)

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
    MyModelHash = session["model_hash"]
    mytable = pd.DataFrame()
    for symbol in MyModelHash.keys():
        mytable = mytable.append(MyModelHash[symbol].df_info)
    mytable.sort_values(by=['symbol'], inplace=True)

    mytable['myindex'] = mytable.index
    html_table = mytable[['myindex', 'symbol', 'name', 'close_price', 'volume', 'fifty_two_week_low', 'fifty_two_week_high', 'eps', 'div_yield', 'sector', 'industry']]

    #TODO: For now hardcoding recommendations while I work on this page
    sames_table = ['AMD', 'NVDA', 'FB', 'APPS', 'MSFT', 'ATVI', 'AEY', 'SQ', 'BB', 'AMAT']
    sames_table = universe[universe['symbol'].isin(sames_table)]
    #sames_table.sort_values(by=['symbol'], inplace=True)
    sames_table['myindex'] = sames_table.index
    sames_table = sames_table[html_table.columns.tolist()]

    serendipity_table = ['JPM', 'NKE', 'BBIO', 'CCL', 'AMC', 'AVCT', 'BAC', 'AAL', 'BBD', 'MUSA']
    serendipity_table = universe[universe['symbol'].isin(serendipity_table)]
    #serendipity_table.sort_values(by=['symbol'], inplace=True)
    serendipity_table['myindex'] = serendipity_table.index
    serendipity_table = serendipity_table[html_table.columns.tolist()]

    ##app.logger.debug("******************** Symbols: {}".format(html_table.to_numpy().tolist()))
    
    # render web page with data
    return render_template('recommend.html', tables=html_table.to_numpy().tolist(), sames=sames_table.to_numpy().tolist(), serendipity=serendipity_table.to_numpy().tolist())


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
