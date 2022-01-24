import numpy as np
import pandas as pd
import datetime

class StockDateValidator:
    """
    A class to validate dates and date ranges for historical data and stock price predictions.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    isValidRange(from_date: datetime.date, to_date: datetime.date, can_be_in_the_future):
        Determines if the given dates are valid, not in the future, and from_date is before to_date
    parseDateRange(aString: str):
        Returns start and end dates for the given date range string.
    """
    def isValidRange(from_date: datetime.date, to_date: datetime.date, can_be_in_the_future = False):
        '''
        INPUT:
        from_date (Date) - the starting date range 
        to_date (Date) - the ending date range
        can_be_in_the_future (Boolean) - if True the date range may occur beyond today or in the future
        
        OUTPUT:
        error_msg - Optional error message if not valid
        boolean - True if the date range is considered valid, false otherwise
        from_date (Date) - Converted from Date
        to_date (Date) - Converted to Date
        
        Description:
        Given a date range if the dates given are valid, not in the future, and from_date is before to_date
        '''
        # First make sure we have valid dates
        newFromDate = datetime.datetime(1900, 1, 1)
        newToDate = datetime.datetime(1900, 1, 1)

        correctFromDate = isinstance(from_date, datetime.date)
        try:
            if (correctFromDate):
                newFromDate = from_date
            elif (isinstance(from_date, np.datetime64)):
                newFromDate = pd.to_datetime(from_date)
            else:
                newFromDate = datetime.datetime.strptime(from_date, '%Y-%m-%d')
            correctFromDate = True
        except:
                correctFromDate = False

        correctToDate = isinstance(to_date, datetime.date)
        try:
            if (correctToDate):
                newToDate = to_date
            elif (isinstance(to_date, np.datetime64)):
                newToDate = pd.to_datetime(to_date)
            else:
                newToDate = datetime.datetime.strptime(to_date, '%Y-%m-%d')
            correctToDate = True
        except:
                correctToDate = False
        
        if (not correctFromDate and not correctToDate):
            return "Invalid from date *and* invalid to date.", False, newFromDate, newToDate
        if (not correctFromDate):
            return "Invalid from date.", False, newFromDate, newToDate
        if (not correctToDate):
            return "Invalid to date.", False, newFromDate, newToDate
        
        if (newToDate <= newFromDate):
            return "Invalid date range, to_date {} must be greater than from_date {}.".format(newToDate, newFromDate), False, newFromDate, newToDate
        
        if (not can_be_in_the_future):
            today = datetime.datetime.today()
            if (newToDate > today):
                return "Invalid date range, to_date {} cannot be in the future or greater than today.".format(newToDate, today.date), False, newFromDate, newToDate
        
        return "", True, newFromDate, newToDate

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

        start = np.datetime64(startStr, 'D')
        end = np.datetime64(endStr, 'D')

        return start, end
