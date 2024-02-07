# -*- coding: utf-8 -*-
"""
Created on Feb 07

@author: DorianGuzman
"""
import pvlib
import pandas as pd
from typing import Tuple
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def normalize_df_by_direction(imputed_df, top_values_df,
                              directions_df, direction='west') -> (pd.DataFrame, pd.DataFrame):
    """
    Normalize DataFrame based on directions.

    Parameters
    ----------
    imputed_df : pd.DataFrame
        The DataFrame with imputed values.
    top_values_df : pd.DataFrame
        The DataFrame with top values.
    directions_df : pd.DataFrame
        The DataFrame with assigned directions.
    direction : str
        The specific direction to consider, should be one of ['west', 'south', 'east'].

    Returns
    -------
    pd.DataFrame
        The normalized DataFrame.
    pd.DataFrame
        Normalization scale to scale back values
    """
    # Extract column names based on the directions
    east_columns = directions_df[directions_df['Direction'] == 'east'].index
    west_columns = directions_df[directions_df['Direction'] == 'west'].index
    south_columns = directions_df[directions_df['Direction'] == 'south'].index

    # Check the direction and assign appropriate columns
    if direction == 'west':
        direction_cols = west_columns
        top_df = top_values_df[direction_cols]
        max_string = top_df.max()
        # Get the normalization factor
        norm_factor = top_df.max().max()
        # Get the normalization scale dataframe
        norm_scale = max_string / norm_factor
        # Define a function to replace values
        def replace_values(x):
            if 0 <= x <= 0.3:
                return 0.25
            elif 0.3 < x <= 0.7:
                return 0.5
            elif 0.7 < x <= 1:
                return 1
            else:
                return x
        # Apply the function to the Series
        norm_scale = norm_scale.apply(replace_values)
        norm_scale = norm_scale.T * norm_factor
        # Normalized dataframe
        norm_df = imputed_df / norm_scale
    elif direction == 'east' or direction == 'south':
        if direction == 'south':
            direction_cols = south_columns
        else:
            direction_cols = east_columns
        top_df = top_values_df[direction_cols]
        max_string = top_df.max()
        # Get the normalization factor
        norm_factor = top_df.max().max()
        # Get the normalization scale dataframe
        norm_scale = max_string / norm_factor
        # Replace values less than 1 with 1
        norm_scale = norm_scale.apply(lambda x: max(1, x)) 
        norm_scale = norm_scale.T * norm_factor
        # Normalized dataframe
        norm_df = imputed_df / norm_scale
    else:
        raise ValueError("Invalid direction. Should be one of ['west', 'south', 'east']")

    return norm_df, norm_scale


def impute_and_replace_negatives(df, max_iter=10, random_state=0):
    """
    Impute missing values and replace negatives with 0 using IterativeImputer.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with missing values.
    max_iter : int, optional
        Maximum number of imputation iterations, by default 10.
    random_state : int, optional
        Seed for reproducibility, by default 0.

    Returns
    -------
    pd.DataFrame
        The imputed and processed DataFrame.
    """
    # Instantiate IterativeImputer
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)

    # Fit and transform the DataFrame to impute missing values
    imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

    # Replace negative values by 0
    imputed_df = imputed_df.applymap(lambda x: max(0, x))

    return imputed_df


def assign_directions(df: pd.DataFrame, start_date: str, end_date: str, timezone: str,
                      range_east: tuple = (0, 12), range_west: tuple = (14, 23),
                      range_center: tuple = (13, 15),
                      top_n: int = 3) -> Tuple[
                          pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign directions based on value distribution for a specified date range.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the original data.
    start_date : str
        The start date for filtering values.
    end_date : str
        The end date for filtering values.
    timezone : str
        The timezone for date localization (e.g., 'Europe/Berlin').
    range_east : tuple, optional
        The range for eastern values, by default (0, 12).
    range_west : tuple, optional
        The range for western values, by default (14, 23).
    range_center : tuple, optional
        The range for central values, by default (13, 15).
    top_n : int, optional
        The number of top values to consider per day and column, by default 3.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three filtered DataFrames ('east_df', 'west_df', 'south_df')
        and the directions DataFrame ('directions_df').

    Example
    -------
    directions_df = assign_directions(df, '2023-07-07', '2023-07-09', 'Europe/Berlin')
    """

    # Convert start and end dates to datetime objects with the specified timezone
    start_date = pd.to_datetime(start_date).tz_localize(timezone)
    end_date = pd.to_datetime(end_date).tz_localize(timezone)

    # Filter values only from the specified date range
    ck_days = df.loc[start_date:end_date]

    # Calculate the median per day for each column
    max_per_hour = ck_days.groupby(ck_days.index.hour).max()

    # Get the highest 'top_n' values per day for each column
    top_values_df = max_per_hour.apply(lambda col: col.nlargest(top_n))

    # Function to determine direction based on value distribution
    def assign_direction(column):
        column = column.dropna()
        values_east = column.index.isin(range(*range_east)).sum()
        values_west = column.index.isin(range(*range_west)).sum()
        values_center = column.index.isin(range(*range_center)).sum()
        if max(values_east, values_west, values_center) == values_east:
            return "east"
        elif max(values_east, values_west, values_center) == values_west:
            return "west"
        else:
            return 'south'

    # Apply the function to each column
    directions = top_values_df.apply(assign_direction)

    # Create a DataFrame with column names as index and directions as values
    directions_df = pd.DataFrame({'Direction': directions})

    # Extract column names based on the directions
    east_columns = directions_df[directions_df['Direction'] == 'east'].index
    west_columns = directions_df[directions_df['Direction'] == 'west'].index
    south_columns = directions_df[directions_df['Direction'] == 'south'].index

    # Filter the DataFrame based on the extracted column names
    east_df = df[east_columns]
    west_df = df[west_columns]
    south_df = df[south_columns]

    return east_df, west_df, south_df, directions_df, top_values_df


def clean_data(df: pd.DataFrame, latitude: float, longitude: float,
              neg_threshold: float = 30, zero_threshold: float = 50) -> pd.DataFrame:
    """
    Clean and preprocess data by filtering clear-sky irradiance, negative values, and zero-dominated columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the original data.
    latitude : float
        The latitude of the location.
    longitude : float
        The longitude of the location.
    neg_threshold : float, optional
        The maximum percentage allowed for negative values, by default 30.
    zero_threshold : float, optional
        The maximum percentage allowed for zero values in columns, by default 50.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame after applying filters.

    Example
    -------
    cleaned_df = clean_data(df, latitude, longitude, neg_threshold=30, zero_threshold=50)
    """

    # Get clearsky
    clearsky = clearsky_get(
        latitude=latitude,
        longitude=longitude,
        times=df.index)

    total_data = df.size

    # Filter out values when clear-sky irradiance is zero
    df = filter_clearsky_irradiance(
        df=df,
        clearsky=clearsky)

    # Filter out negative values and NaNs
    # Call the filter
    df, neg_col_list = filter_negative_values(
        df=df,
        threshold=neg_threshold)

    # Filter out String with only Zero values
    df_filtered = zero_filter(
        df=df,
        neg_col_list=neg_col_list,
        total_data=total_data,
        zero_threshold=zero_threshold)

    return df_filtered


def zero_filter(df: pd.DataFrame, neg_col_list: list, total_data: int,
                zero_threshold: float = 50) -> pd.DataFrame:
    """
    Filter out columns with a percentage of zero values exceeding the specified threshold.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the original data.
    total_data : int
        The total number of data points in the original DataFrame.
    zero_threshold : float, optional
        The threshold percentage for columns with only zero values, by default 50.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame with columns exceeding the zero threshold removed.

    Example
    -------
    df_filtered = zero_filter(df, total_data)
    """

    # Check if 50% of the values in each column are zero
    percentage_zeros = (df == 0).mean() * 100
    threshhold_zeros = percentage_zeros[percentage_zeros > zero_threshold].index

    # Filter out columns with only zero values
    df_zero_filtered = df.drop(columns=threshhold_zeros)

    # Print the report
    print("\nReport After all the Filters:")
    print("Columns with {}% or more only zero values:".format(zero_threshold))
    print(threshhold_zeros.values)

    # Count the number of columns removed
    columns_removed = list(set(df.columns) - set(df_zero_filtered.columns))
    print("Number of Columns with Zero Removed: {}".format(len(columns_removed)))

    # Print additional information
    print("\n\nTotal Columns Removed from Filters: {}".format(
        len(neg_col_list) + len(columns_removed)))
    total_data_deleted = round(((total_data - df_zero_filtered.count().sum()) / total_data) * 100, 2)
    print("Deleted Columns: {}".format(neg_col_list+columns_removed))
    print("\n% of Total Data Deleted (NaN and Negative): {:.2f}%".format(total_data_deleted))

    return df_zero_filtered


def filter_negative_values(df: pd.DataFrame, threshold: float = 30) -> pd.DataFrame:
    """
    Filter out negative values and columns exceeding the specified threshold percentage.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the original data.
    threshold : float, optional
        The threshold percentage for the total negative values in a column, by default 30.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame with negative values and columns exceeding the threshold removed.

    Example
    -------
    df_filtered = filter_negative_values(df)
    """

    # Before filtering
    total_data_f1 = df.size

    # Report
    report = pd.DataFrame(columns=['Column', 'Total Negative (%)'])
    neg_col_list = []

    for col in df.columns:
        before_negative_percentage = (df[col] < 0).sum() / df[col].count() * 100
        if before_negative_percentage > 0:
            report = report.append({
                'Column': col,
                'Total Negative (%)': round(before_negative_percentage, 2),
            }, ignore_index=True)
            if before_negative_percentage > threshold:
                neg_col_list.append(col)

    # Print the report
    print("\nNegative and NaN Filter Report:")
    print(report)

    # Exclude specified columns
    df_neg_filter = df.drop(columns=neg_col_list, errors='ignore')

    # Filter out negative values and NaN
    df_neg_filter = df_neg_filter[df_neg_filter >= 0]

    # After filtering
    total_data_after_neg = df_neg_filter.count().sum()

    # Calculate the percentage of total data deleted
    percentage_data_deleted = round(((total_data_f1 - total_data_after_neg) / total_data_f1) * 100, 2)

    # Generate the final report
    final_report = f"\n% of Total Data Deleted (NaN and Negative): {percentage_data_deleted:.2f}%\nDeleted Columns: {neg_col_list}"

    # Print the final report
    print(final_report)

    return df_neg_filter, neg_col_list


def filter_clearsky_irradiance(df: pd.DataFrame, clearsky: pd.Series) -> pd.DataFrame:
    """
    Filter out values from the DataFrame where clear-sky irradiance is zero.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the original data.
    clearsky : pd.Series
        The clear-sky irradiance values corresponding to the times in the DataFrame.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame where clear-sky irradiance is greater than zero.

    Raises
    ------
    ValueError
        If the lengths of the DataFrame and clearsky Series do not match.
    
    Notes
    -----
    This function also prints a report on the percentage of total data removed.
    
    Example
    -------
    df_filtered, report = filter_clearsky_irradiance(df, clearsky)
    """
    # Check if the lengths match
    if len(df) != len(clearsky):
        raise ValueError("Lengths of DataFrame and clearsky Series do not match.")

    # Before filtering
    total_data = df.size

    # Filter values where clear-sky irradiance is higher than zero
    df_filtered = df[clearsky > 0]

    # After filtering
    total_data_after_ck = df_filtered.count().sum()

    # Calculate the percentage of total data removed
    percentage_removed = ((total_data - total_data_after_ck) / total_data) * 100

    # Generate the report
    report = f"\nClear Sky Filter Report:\n% of Total Data Removed From Full Dataset: {percentage_removed:.2f}%\nTotal Data Removed: {total_data - total_data_after_ck}"

    # Print the report
    print(report)

    return df_filtered


def clearsky_get(latitude: float, longitude: float, times: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate clear-sky Global Horizontal Irradiance (GHI) using Ineichen clear-sky model.

    Parameters
    ----------
    latitude : float
        The latitude of the location.
    longitude : float
        The longitude of the location.
    times : pd.DatetimeIndex
        The time index for which to calculate clear-sky GHI.

    Returns
    -------
    pd.Series
        A Series containing clear-sky GHI values for the specified times.
    """
    # Create a location object
    location = pvlib.location.Location(latitude=latitude, longitude=longitude)

    # Calculate clear-sky GHI using Ineichen clear-sky model
    clearsky = location.get_clearsky(times=times)

    # Filter out values when actual irradiance is lower than 0
    clearsky_ghi = clearsky[clearsky['dhi'] >= 0]['dhi']

    return clearsky_ghi
