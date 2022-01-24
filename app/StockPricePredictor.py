import numpy as np
import pandas as pd
import datetime
import logging

from StockModel import StockModel
from StockDateValidator import StockDateValidator

class StockPricePredictor:
    """
    A class to encapsulate getting stock price predictions using a StockModel

    ...

    Attributes
    ----------
    model_hash : dict
        StockModel dictionary cache
    """
    def __init__(self, model_hash: dict):
        """
        Constructs all the necessary attributes for the StockPricePredictor object.

        Parameters
        ----------
            model_hash : dict
                StockModel dictionary cache
        """
        self.model_hash = model_hash
    
    def areValidSymbols(self, symbols: list):
        '''
        INPUT:
        symbols - (list) List of stock tickers to evaluate if it is a valid list and all symbols
                        have an entry in the prediction model hash
        
        OUTPUT:
        error_msg - Optional error message when applicable, the reason why the given symbols list is not valid
        bool_ret - if False the given list is not valid
        
        Description:
        This validates the given list of symbols to make sure it is a valid list of symbols and each
        symbol has a corresponding entry in the model_hash dictionary 
        '''
        # First make sure we have a valid ticker list
        if (not symbols or len(symbols) == 0):
            return "Error: invalid or empty list of symbols passed.", False
        # Next make sure we have a trained model for the given tickers
        for symbol in symbols:
            if not symbol in self.model_hash.keys():
                return "Error: no trained model for symbol '{}'.".format(symbol), False
            
        return "", True

    def evaluateModels(self, symbols: list):
        '''
        INPUT:
        symbols - (array) List of stock tickers to evaluate prediction models for
        
        OUTPUT:
        error_msg - Optional error message when applicable, the reason why the given symbols list is not valid
        bool_ret - if False the given list is not valid
        
        Description:
        This iterates the given symbols and outputs the model evaluate_model function output
        '''
        # Validate the symbols given
        error_msg, bool_ret = self.areValidSymbols(symbols)
        if (not bool_ret):
            return error_msg, bool_ret
        # Finally use the models to return the output of the evaluate_model function
        for symbol in symbols:
            ##logging.debug("*** Symbol: {}\n".format(symbol))
            tm = self.model_hash[symbol]
            tm.evaluate_model()
        return "", True
        
    def getStockPredictions(self, symbols: list, from_date: datetime.date, to_date: datetime.date, max_days_to_predict = 100):
        '''
        INPUT:
        symbols - (list) List of stock tickers to get stock prediction data for
        from_date - (datetime.date) The starting date range to get data for 
        to_date - (datetime.date) The ending date range to get data for 
        max_days_to_predict (int) - maximum number of days to predict into the future (sanity check)
        
        OUTPUT:
        error_msg - Optional error message when applicable, if empty the data frame is valid
        df_ret - Pandas data frame containing columns Symbol, Date, Open, High, Low, Close, Adj Close, and Adj Close Next Day
        
        Description:
        This is the query interface function that accepts a list of dates and a list of ticker symbols, 
        and outputs the predicted stock prices for each of those stocks on the given dates. 
        Note that the query dates passed in must be after the training date range, and ticker symbols 
        must be a subset of the ones trained on.
        '''
        # Initialize the return data frame as empty
        df_ret = pd.DataFrame()
        # Validate the symbols given
        error_msg, bool_ret = self.areValidSymbols(symbols)
        if (not bool_ret):
            return error_msg, df_ret
        # Finally use the models to predict for the given tickers in the date range
        for symbol in symbols:
            tm = self.model_hash[symbol]
            # call this to get valid date objects newFromDate, newToDate
            error_msg, bool_val, newFromDate, newToDate = StockDateValidator.isValidRange(from_date, to_date, True)
            # make sure the given date range is after the training date range
            if (newFromDate <= tm.last_trained_date):##tm.to_date):
                return "Error: prediction dates passed in starting on '{}' must begin after the training date range from '{}' to '{}'.".format(newFromDate, tm.from_date, tm.last_trained_date), df_ret
            # if we have a limit of how many days into the future to predict make sure we do not exceed that limit
            delta = newToDate - newFromDate
            if (max_days_to_predict > 0 and delta.days > max_days_to_predict):
                return "Error: given date range from '{}' to '{}' exceeds maximum number of days to predict {} by {} day(s).".format(newFromDate, newToDate, max_days_to_predict, (delta.days - max_days_to_predict)), df_ret
            # iterate over our date range one day at a time
            delta = datetime.timedelta(days=1)
            # start with one day beyond our training data or our test data because we build
            # off the last trained data row N
            current_date = tm.last_trained_date + delta
            # Initiate data frame to the last element of the training data
            df_curr = tm.df[tm.df['Date'] > tm.last_trained_date].head(1)
            # Iterate one day past because that is how we can predict to day N, by going to day N+1
            while current_date <= (newToDate + delta):
                # Predict using the model
                df_next = tm.predict(current_date, df_curr)
                # First iteration we are getting a prediction on the last train/test data, so we are getting a better N-1
                if (current_date == (tm.last_trained_date + delta)):
                    df_curr = df_next
                else:
                    df_curr = df_curr.append(df_next)
                current_date += delta
            # Remove day N+1
            df_curr.drop(df_curr.tail(1).index,inplace=True)
            # Remove days prior to N in the case of where we are predicting beyond the end of our test data
            df_curr = df_curr[df_curr['Date'] >= newFromDate]
            # Append our current symbol results to the overall df_ret dataframe
            df_ret = df_ret.append(df_curr, ignore_index=True)
        
        return "", df_ret