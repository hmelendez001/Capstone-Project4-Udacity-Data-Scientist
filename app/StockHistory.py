import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import logging

from StockDateValidator import StockDateValidator

class StockHistory:
    """
    A class to manage stock history data, caching data appropriately to minimize calls to the Yahoo! Finance API.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    getFromCache(symbol: str, from_date: datetime.date, to_date: datetime.date):
        Given a valid stock symbol ticker and date range, check if we have already cached 
        this information and if so return it. We have refactored this function because 
        we may use a database to cache this in the future, but for now are caching using a
        dictionary: stock_hash
    """
    def __init__(self, stock_hash: dict, model_hash: dict) -> None:
        '''
        INPUT:
        stock_hash (dict) - Dictionary symbol cache 
        model_hash (dict) - Dictionary StockModel cache 
        
        OUTPUT:
        None
        
        Description:
        The init method or class constructor
        '''
        self.stock_hash = stock_hash
        self.model_hash = model_hash

    def dropModelFromCache(self, symbol: str):
        '''
        INPUT:
        symbol - Input stock ticker to drop from model_hash
        
        OUTPUT:
        model_hash (dict) - Updated model_hash dictionary
        
        Description:
        Given a valid stock symbol ticker remove it from the dictionary: model_hash
        '''
        del self.model_hash[symbol]
        return self.model_hash

    def getFromCache(self, symbol: str, from_date: datetime.date, to_date: datetime.date):
        '''
        INPUT:
        symbol - Input stock ticker to get historical data for
        from_date - (date) the starting date range to get data for 
        to_date - (date) the ending date range to get data for 
        
        OUTPUT:
        error_msg (str) - Optional error message when applicable, if empty the data frame is valid
        bool_val (Boolean) - If True then we found a valid DataFrame in our cache for the given parameters
        df_ret (Pandas DataFrame) - DataFrame containing columns Date, Open, High, Low, Close, and Adj Close
        
        Description:
        Given a valid stock symbol ticker and date range, check if we have already cached 
        this information and if so return it. We have refactored this function because 
        we may use a database to cache this in the future, but for now are caching using a
        dictionary: stock_hash
        '''
        # Initialize empy data frame to return by default
        df_ret = pd.DataFrame()
        
        # Get today's date because we need to check our cache first
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        key = today + "|" + symbol
        
        # Check if stock_hash already has this. If the data frame cached is None the set the appropriate error message
        if key in self.stock_hash.keys():
            df_ret = self.stock_hash[key]
            if (len(df_ret) == 0):
                return "Invalid symbol '{}' passed.".format(symbol), True, df_ret
            # Even if we have this symbol in our cache for today make sure the date range is within our cached date range
            if ((from_date < df_ret.Date.min()) or (to_date > df_ret.Date.max())):
                return "Cached dates [{} to {}] outside the given range [{} to {}]".format(df_ret.Date.min(), df_ret.Date.max(), from_date, to_date), True, df_ret
            return "", True, df_ret[(df_ret['Date'] >= from_date) & (df_ret['Date'] <= to_date)]
        return "", False, df_ret

    def getStockHistory(self, symbol: str, from_date: datetime.date, to_date: datetime.date):
        '''
        INPUT:
        symbol - Input stock ticker to get historical data for
        from_date - (date) the starting date range to get data for 
        to_date - (date) the ending date range to get data for 
        
        OUTPUT:
        error_msg - Optional error message when applicable, if empty the data frame is valid
        df - Pandas data frame containing columns Date, Open, High, Low, Close, Adj Close
        
        Description:
        Given a valid stock symbol ticker and date range, check if we have already cached 
        this information in the stock_hash dictionary, if not, call the Yahoo! Finance API
        to get this data, cached it, and return the results.
        '''
        # First make sure we have a valid ticker string
        if (not symbol):
            return "Invalid symbol string passed.", pd.DataFrame()
        
        # Now check for a valid date range
        error_msg, bool_val, from_date, to_date = StockDateValidator.isValidRange(from_date, to_date)
        if (not bool_val):
            return error_msg, None
        
        # See if we already have this in our cache
        error_msg, bool_val, df_ret = self.getFromCache(symbol, from_date, to_date)
        if (bool_val):
            return error_msg, df_ret
        
        # Call the API: Yahoo! Finance is not inclusive of the date ranges to subtract and add one day accordingly
        df_ret = yf.download(symbol, from_date + datetime.timedelta(days=-1), to_date + datetime.timedelta(days=1))
        df_ret = df_ret.reset_index()

        if (len(df_ret) == 0):
            error_msg = "Invalid symbol '{}' passed.".format(symbol)
        
        # Get today's date because we need to check our cache first
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        key = today + "|" + symbol
        
        # Store it in the hash for next time we call it
        if key in self.stock_hash.keys():
            del self.stock_hash[key]
        self.stock_hash[key] = df_ret

        return error_msg, df_ret

    def getStockHistories(self, symbols: list, from_date: datetime.date, to_date: datetime.date, CreateStockModel, test_size = 0.3, random_state = 42):
        '''
        INPUT:
        symbols (list) - List of stock tickers to get historical data for
        from_date (date) - The starting date range to get data for 
        to_date (date) - The ending date range to get data for 
        CreateStockModel (func) - Function to return the StockModel constructor being used
        test_size (numeric) - Optional test size, can be a float in the (0, 1) range to indicate a percentage like .01 = 1%, 
                            or a number that is at least 1 but less than the total size. 
                            We default to 1 because we want to train with most of the data.
        random_state (numeric) - Optional random state
        
        OUTPUT:
        error_msg (str) - Optional error message when applicable, if empty the data frame is valid
        bool_ret (Boolean) - If True then the function ran successfully, else the error is contained in error_msg
        model_hash (dict) - The model hash dictionary
        
        Description:
        This is the training interface function that accepts a data range (start_date, end_date) and 
        a list of ticker symbols (e.g. GOOG, AAPL), and builds a model of stock behavior. 
        The code will read the desired historical prices from the Yahoo! Finance data source API, if not cached already.
        '''
        # First make sure we have a valid ticker list
        if (len(symbols) == 0):
            return "Error: invalid or empty list of symbols passed.", False
        # For each symbol train and cache a model
        for symbol in symbols:
            error_msg, df_ret = self.getStockHistory(symbol, from_date, to_date)
            if (len(error_msg) > 0 or len(df_ret) == 0):
                return error_msg, False, self.model_hash
            # call this to get valid date objects newFromDate, newToDate
            error_msg, bool_val, newFromDate, newToDate = StockDateValidator.isValidRange(from_date, to_date)
            # Add column Adj Close Next Day
            df_symbol = df_ret.copy()
            # Add Symbol column first, it not already there
            if not 'Symbol' in df_symbol.columns:
                df_symbol.insert(0, 'Symbol', symbol)
            # Take next day's Adj Close and put in in the current Adj Close Next Day column
            df_symbol['Adj Close Next Day'] = df_symbol['Adj Close'].shift(-1)
            # Backfill the nth day
            df_symbol = df_symbol.ffill().bfill()

            lm_symbol_model = CreateStockModel(symbol, df_symbol, newFromDate, newToDate)
            if test_size < 1:
                test_size = int(test_size * df_symbol.shape[0])
            lm_symbol_model.fit(test_size, random_state)
            
            self.model_hash[symbol] = lm_symbol_model
            
        return "", True, self.model_hash
