import numpy as np
import pandas as pd
import datetime
from sklearn.pipeline import make_pipeline

class StockModel:
    """
    A base class to encapsulate a trained model that uses any implemented algo and the date range for which it was trained.

    ...

    Attributes
    ----------
    from_date : date
        beginning date range
    to_date : date
        ending date range
    df_info : Pandas.DataFrame
        Additional symbol information dataframe, columns: 
        'symbol', 'name', 'close_price', 'volume', 'fifty_two_week_low', 'fifty_two_week_high', 'eps', 'div_yield', 'sector', 'industry', 'begin_train', 'end_train'
    model : sklearn.pipeline.Pipeline
        the trained model

    Methods
    -------
    getFromCache(symbol: str, from_date: datetime.date, to_date: datetime.date):
        Given a valid stock symbol ticker and date range, check if we have already cached 
        this information and if so return it. We have refactored this function because 
        we may use a database to cache this in the future, but for now are caching using a
        dictionary: stock_hash
    """
    def __init__(self, symbol: str, df: pd.DataFrame, from_date: datetime.date, to_date: datetime.date, df_info: pd.DataFrame) -> None:
        """
        Constructs all the necessary attributes for the base TrainedModel object.

        Parameters
        ----------
            symbol : str
                The stock ticker symbol, e.g. GOOGL, MSFT, etc.
            df : Pandas DataFrame
                The underlying historical DataFrame
            from_date : Date
                beginning date range
            to_date : Date
                ending date range
            df_info : Pandas.DataFrame
                Additional symbol information dataframe, columns: 
                'symbol', 'name', 'close_price', 'volume', 'fifty_two_week_low', 'fifty_two_week_high', 'eps', 'div_yield', 'sector', 'industry', 'begin_train', 'end_train'
            model_hash : dict
                StockModel dictionary cache
        """
        self.name = ""
        self.symbol = symbol
#        self.model_hash = model_hash
        self.df = df
        self.df_info = df_info
        if ('Date' in df.columns):
            self.df = df.sort_values(by=['Date'])
        self.modelAdjCloseNextDay = None
        self.modelOpen = None
        self.modelHigh = None
        self.modelLow = None
        self.modelClose = None
        self.from_date = from_date
        self.to_date = to_date
        # For now set to end date and empty test data frames, but this will be updated after function fit runs
        self.last_trained_date = to_date
        self.X_testAdjCloseNextDay = pd.DataFrame()
        self.y_testAdjCloseNextDay = pd.DataFrame()
 
    def fit(self, test_size: int, random_state: int):
        """
        Fits the inputs using TrainedModel model, otherwise known as training the model.

        Parameters
        ----------
            test_size : numeric
                How much of the training data gets set aside for test data
            random_state : numeric
                Numeric that allows us to repeat a "random value"
        """
        raise ValueError("*** ERROR: Class {} does not implement fit method.", self.name)

    def predict(self, current_date: datetime.date, df_curr: pd.DataFrame):
        """
        Make the prediction for the given current_date using the TrainedModel object.
        We account for scenarios where tests are run for data within the range of our test data, 
        meaning we have the reported Open, High, Low, Close, Adj Close already, *or* the case 
        where we are asked to predict beyond our test data.

        Parameters
        ----------
            current_date : datetime.date
                The date being predicted
            df_curr : pandas.DataFrame
                The DataFrame containing the inputs for the previous date used to predict current_date data
                
        Returns
        -------
            df_next : pandas.DataFrame
                Resulting DataFrame containing current_date predictions
        """
        raise ValueError("*** ERROR: Class {} does not implement predict method.", self.name)
    
    def evaluate_model(self, X_test = pd.DataFrame(), Y_test = pd.DataFrame()):
        """
        Evaluates the given model with the given X test data and expected Y test data
        Parameters
        ----------
        X_test: pandas.DataFrame
            Optional X input test data, if not given use the test data we have self.X_testAdjCloseNextDay
        Y_test: pandas.DataFrame
            Optional Y output test data, if not given use the test data we have self.y_testAdjCloseNextDay
        Returns
        -------
        None
        Examples
        --------
        >>> evaluate_model()
        >>> evaluate_model(X_test, Y_test)

        """
        raise ValueError("*** ERROR: Class {} does not implement evaluate_model method.", self.name)
