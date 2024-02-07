# -*- coding: utf-8 -*-
"""
Created on Feb 07

@author: DorianGuzman
"""

import os
import pandas as pd
from pytz import timezone


def read_data(filename: str) -> pd.DataFrame:
    """
    Read data from a CSV file and preprocess it for analysis.

    Parameters
    ----------
    filename : str
        The path to the CSV file to read.

    Returns
    -------
    pd.DataFrame
        A DataFrame with preprocessed data for analysis.
    """
    
    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Read the CSV file
    df = pd.read_csv(filename, delimiter=';')

    # Convert the 'Datum' column to a datetime object and set it as the index
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y %H:%M')
    europe_timezone = timezone('Europe/Berlin')  # Assuming Central Europe time zone
    df['Datum'] = df['Datum'].dt.tz_localize(europe_timezone)
    df.set_index('Datum', inplace=True)

    # Rename the columns by adding 'String' to each column name
    df.columns = [f'String{col}' for col in df.columns]

    # Replace ',' by '.'
    # Some values have , instead of ., therefore this has to be 
    # replaced, and converted back to numerical values
    # Otherwise, values that cannot be converted to numeric will be
    # replaced by NaN. 
    df.replace(',', '.', regex=True, inplace=True)

    # Convert columns to numeric
    df = df.apply(pd.to_numeric, downcast="float", errors='coerce').round(2)

    return df


