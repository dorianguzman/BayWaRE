# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24

@author: DorianGuzman
"""
import pandas as pd
import os


def read_data(filename: str)->pd.DataFrame:
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

    # Define the data types for specific columns
    dtype_mapping = {
        'data_date': 'str',
        'wind_speed': 'float32',
        'tmp_amb': 'float32',
        'poa': 'float32',
        'tmp_cell': 'float32',
        'ac_power': 'float32',
        'rain_mm': 'float32'
    }

    # Read the CSV file into a DataFrame with predefined data types
    df = pd.read_csv(filename, dtype=dtype_mapping, parse_dates=['data_date'])
    # Rename columns
    df = df.rename(columns={'data_date': 'datetime'})
    
    # Set the 'datetime' column as the DataFrame index
    df.set_index('datetime', drop=True, inplace=True)
    
    # Convert 'ac_power' to watts
    df['ac_power'] *= 1000

    return df


