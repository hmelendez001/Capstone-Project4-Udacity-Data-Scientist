import numpy as np
import pandas as pd
import datetime
import logging

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score

from StockModel import StockModel
from StockDateValidator import StockDateValidator

class StockModelLinear(StockModel):
    """
    A class to encapsulate a trained model that uses Linear Regression and the date range for which it was trained.

    ...

    Attributes
    ----------
    model : sklearn.pipeline.Pipeline
        the trained model
    from_date : date
        beginning date range
    to_date : date
        ending date range
    """
    def __init__(self, symbol: str, df: pd.DataFrame, from_date: datetime.date, to_date: datetime.date, df_info: pd.DataFrame):
        """
        Constructs all the necessary attributes for the TrainedModelLinear object.

        Parameters
        ----------
            symbol : str
                The stock ticker symbol, e.g. GOOGL, MSFT, etc.
            df : Pandas.DataFrame
                The underlying historical DataFrame
            from_date : datetime.date
                beginning date range
            to_date : datetime.date
                ending date range
            df_info : Pandas.DataFrame
                Additional symbol information dataframe, columns: 
                'symbol', 'name', 'close_price', 'volume', 'fifty_two_week_low', 'fifty_two_week_high', 'eps', 'div_yield', 'sector', 'industry', 'begin_train', 'end_train'
        """
        self.symbol = symbol
        self.df = df
        self.df_info = df_info
        if ('Date' in df.columns):
            self.df = df.sort_values(by=['Date'])
        self.modelAdjCloseNextDay = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        self.modelOpen = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        self.modelHigh = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        self.modelLow = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        self.modelClose = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        self.from_date = from_date
        self.to_date = to_date
        # For now set to end date and empty test data frames, but this will be updated after function fit runs
        self.last_trained_date = to_date
        self.X_testAdjCloseNextDay = pd.DataFrame()
        self.y_testAdjCloseNextDay = pd.DataFrame()
    
    def fit(self, test_size: int, random_state: int):
        """
        Fits the inputs using TrainedModelLinear model, otherwise known as training the model. Initially we started 
        using train_test_split to split our training and test data but then realized we did not want to split it
        randomly but rather in order since we want to train with most of the data and we want to track what the 
        last N Date is we trained for when we start testing the effectiveness of the model.

        Parameters
        ----------
            test_size : numeric
                How much of the training data gets set aside for test data
            random_state : numeric
                Numeric that allows us to repeat a "random value"
        """
        # Set the last_trained_date based on the test_size index - 1
        self.last_trained_date = self.df.loc[self.df.tail(test_size).index - 1, 'Date'].values[0]
        self.last_trained_date = datetime.datetime.strptime(np.datetime_as_string(self.last_trained_date,unit='D'), '%Y-%m-%d')

        #Split into explanatory and response variables for Adj Close Next Day
        X2_AdjCloseNextDay = self.df[['Open', 'High', 'Low', 'Close', 'Adj Close']]
        y2_AdjCloseNextDay = self.df['Adj Close Next Day']

        #Split into train and test, then fit the model for Adj Close Next Day
        ##X_trainAdjCloseNextDay, X_testAdjCloseNextDay, y_trainAdjCloseNextDay, y_testAdjCloseNextDay = train_test_split(X2_AdjCloseNextDay, y2_AdjCloseNextDay, test_size = test_size, random_state = random_state) 
        X_trainAdjCloseNextDay = X2_AdjCloseNextDay.drop(X2_AdjCloseNextDay.tail(test_size).index)
        y_trainAdjCloseNextDay = y2_AdjCloseNextDay.drop(y2_AdjCloseNextDay.tail(test_size).index)
        
        # Save test data for optional call to function evaluate model
        self.X_testAdjCloseNextDay = X2_AdjCloseNextDay.tail(test_size)
        self.y_testAdjCloseNextDay = y2_AdjCloseNextDay.tail(test_size)
        
        # Run the model fit for Adj Close Next Day
        self.modelAdjCloseNextDay.fit(X_trainAdjCloseNextDay, y_trainAdjCloseNextDay)

        #Split into explanatory and response variables for Open
        X2_Open = self.df[['High', 'Low', 'Close', 'Adj Close']]
        y2_Open = self.df['Open']

        #Split into train and test, then fit the model for Open
        ##X_trainOpen, X_testOpen, y_trainOpen, y_testOpen = train_test_split(X2_Open, y2_Open, test_size = test_size, random_state = random_state) 
        X_trainOpen = X2_Open.drop(X2_Open.tail(test_size).index)
        y_trainOpen = y2_Open.drop(y2_Open.tail(test_size).index)
        self.modelOpen.fit(X_trainOpen, y_trainOpen)

        #Split into explanatory and response variables for High
        X2_High = self.df[['Open', 'Low', 'Close', 'Adj Close']]
        y2_High = self.df['High']

        #Split into train and test, then fit the model for High
        ##X_trainHigh, X_testHigh, y_trainHigh, y_testHigh = train_test_split(X2_High, y2_High, test_size = test_size, random_state = random_state) 
        X_trainHigh = X2_High.drop(X2_High.tail(test_size).index)
        y_trainHigh = y2_High.drop(y2_High.tail(test_size).index)
        self.modelHigh.fit(X_trainHigh, y_trainHigh)

        #Split into explanatory and response variables for Low
        X2_Low = self.df[['Open', 'High', 'Close', 'Adj Close']]
        y2_Low = self.df['Low']

        #Split into train and test, then fit the model for Low
        ##X_trainLow, X_testLow, y_trainLow, y_testLow = train_test_split(X2_Low, y2_Low, test_size = test_size, random_state = random_state) 
        X_trainLow = X2_Low.drop(X2_Low.tail(test_size).index)
        y_trainLow = y2_Low.drop(y2_Low.tail(test_size).index)
        self.modelLow.fit(X_trainLow, y_trainLow)

        #Split into explanatory and response variables for Close
        X2_Close = self.df[['Open', 'High', 'Low', 'Adj Close']]
        y2_Close = self.df['Close']

        #Split into train and test, then fit the model for Close
        ##X_trainClose, X_testClose, y_trainClose, y_testClose = train_test_split(X2_Close, y2_Close, test_size = test_size, random_state = random_state) 
        X_trainClose = X2_Close.drop(X2_Close.tail(test_size).index)
        y_trainClose = y2_Close.drop(y2_Close.tail(test_size).index)
        self.modelClose.fit(X_trainClose, y_trainClose)

    def predict(self, current_date: datetime.date, df_curr: pd.DataFrame):
        """
        Make the prediction for the given current_date using the TrainedModelLinear object.
        We account for scenarios where tests are run for data within the range of our test data, 
        meaning we have the reported Open, High, Low, Close, Adj Close already, *or* the case 
        where we are asked to predict beyond our test data.

        Parameters
        ----------
            current_date : datetime.date
                The date being predicted
            df_curr : Pandas.DataFrame
                The DataFrame containing the inputs for the previous date used to predict current_date data
                
        Returns
        -------
            df_next : Pandas.DataFrame
                Resulting DataFrame containing current_date predictions
        """
        # Handle the case where the date we want to predict falls outside of our test data so 
        # we have to predict the Open, High, Low, Close, Adj Close for the date in question
        # in order to set up current_date + 1 data, otherwise, we can simply use the existing
        # test data values to predict current_date + 1
        if (current_date > self.to_date):
            # Get the inputs we need to make our prediction
            df_next = df_curr.tail(1).copy()

            X_AdjCloseNextDay = df_next[['Open', 'High', 'Low', 'Close', 'Adj Close']]
            y_AdjCloseNextDay = self.modelAdjCloseNextDay.predict(X_AdjCloseNextDay)
            
            # Set previous day Adj Close Next Day
            df_next.iloc[-1, df_curr.columns.get_loc('Adj Close Next Day')] = y_AdjCloseNextDay[0]
            
            # Now predict all of our indicators which will be used when predicting since in this case
            # current_date is beyond the size of our existing dataset which ends on self.to_date
            X_Open = df_next[[ 'High', 'Low', 'Close', 'Adj Close']]
            y_Open = self.modelOpen.predict(X_Open)

            X_High = df_next[['Open', 'Low', 'Close', 'Adj Close']]
            y_High = self.modelHigh.predict(X_High)

            X_Low = df_next[['Open', 'High', 'Close', 'Adj Close']]
            y_Low = self.modelLow.predict(X_Low)

            X_Close = df_next[['Open', 'High', 'Low', 'Adj Close']]
            y_Close = self.modelClose.predict(X_Close)

            # Add our new record for current_date that uses all of our predicted data
            # Notice we don't fill in Adj Close Next Day as that gets set when you run 
            # for current_date + 1
            df_ret = df_next.append({'Symbol': self.symbol, 'Date': current_date, 'Open': y_Open[0], 'High': y_High[0], 'Low': y_Low[0], 'Close': y_Close[0], 'Adj Close': y_AdjCloseNextDay[0], 'Adj Close Next Day': np.nan}, ignore_index=True)
        else:
            df_ret = self.df[self.df['Date'] == current_date].copy()
            # Get the inputs we need to make our prediction
            df_next = df_curr.tail(1).copy()

            X_AdjCloseNextDay = df_next[['Open', 'High', 'Low', 'Close', 'Adj Close']]
            y_AdjCloseNextDay = self.modelAdjCloseNextDay.predict(X_AdjCloseNextDay)
            # Set previous day Adj Close Next Day
            df_ret.iloc[:, df_ret.columns.get_loc('Adj Close Next Day')] = y_AdjCloseNextDay[0]
            
        return df_ret
    
    def evaluateModel(self, X_test = pd.DataFrame(), Y_test = pd.DataFrame()):
        """
        Evaluates the given model with the given X test data and expected Y test data
        Parameters
        ----------
        X_test: Pandas.DataFrame
            Optional X input test data, if not given use the test data we have self.X_testAdjCloseNextDay
        Y_test: Pandas.DataFrame
            Optional Y output test data, if not given use the test data we have self.y_testAdjCloseNextDay
        Returns
        -------
        results (dict) - Dictionary of relevant scores
        Examples
        --------
        >>> evaluate_model()
        >>> evaluate_model(X_test, Y_test)

        """
        # Use our object test values if none given
        if len(X_test) == 0:
            X_test = self.X_testAdjCloseNextDay
        if len(Y_test) == 0:
            Y_test = self.y_testAdjCloseNextDay
        
        # run the model predictions to evaluate performance
        y_pred = self.modelAdjCloseNextDay.predict(X_test)
        #Rsquared and y_test
        length_y_test = len(Y_test)#num in Y_test

        rsquared_score = -1
        if (length_y_test < 2):
            logging.debug("    The r-squared score of the model is NOT calculated for sample size less than 2: {} value(s).".format(length_y_test))
        else:
            rsquared_score = r2_score(Y_test, y_pred)#r2_score
            logging.debug("    The r-squared score of the model {:.2f} on {} values.".format(rsquared_score, length_y_test))
       
        actuals = Y_test
        predicted = y_pred
        ##plt.scatter(actuals, predicted,  color='Darkblue')
        ##plt.xlabel("Actual Price")
        ##plt.ylabel("Predicted Price")
        ##plt.show()
        x2 = actuals.mean()
        y2 = predicted.mean()
        accuracy = x2/y2
        results = { 'R²|R-squared Score, closer to 1.00 the better': rsquared_score
            , 'Tested Values|Number of data points tested to determine model accuracy': length_y_test
            , 'Accuracy|Model accuracy based on the mean ratios of actual prices over predicted prices, closer to 1.00 the better, higher than 1.00 we are under valueing': accuracy
            , 'MAE|Mean Absolute Error or MAE measures the average magnitude of the errors in a set of predictions, without considering their direction, the closer to 0.00 the better': metrics.mean_absolute_error(actuals, predicted)
            , 'MSE|Mean Squared Error or MSE is the quadratic scoring rule that also measures the average magnitude of the error, the closer to 0.00 the better': metrics.mean_squared_error(actuals, predicted)
            , 'RMSE|Root Mean Squared Error or RMSE measures the average of the squares of the errors, the average squared difference between the estimated values and the actual value, the closer to 0.00 the better': np.sqrt(metrics.mean_squared_error(actuals, predicted))
        }
        return results, actuals, predicted
