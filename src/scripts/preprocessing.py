"""
Fugit nction for the task 0, will show numerical and categorical columns
Will also reduce the size of floats to increase performances
Args : df -> the dataframe to be modified
Return : df -> the modified dataframe
"""
import pandas as pd
import numpy as np

def get_types(df) :

    print(df.dtypes)

def get_numerical(df):
    # store numerical columns in an array
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return numeric_columns

def get_categorical(df):
    # store categorical columns in an array -> we saw with df.dtypes that non numerical columns are "object"
    categorical_columns = df.select_dtypes(include=[object]).columns
    return categorical_columns

def downcast_floats(df,target=32):
    # Reduce float size to increase performance and reduce dataframe size
    # Target type can be ajusted from float32 to any that suits
    for col in df.select_dtypes(include=[np.float64]).columns:
        if target == 16:
            df[col] = df[col].astype(np.float16)
        elif target == 32:
            df[col] = df[col].astype(np.float32)
        else:
            print("Cannot proceed to upcasting, wrong value in argument, must be 32 or 16")
    return df

def removal_of_duplicates(df_params):
    """
    This function takes a DataFrame as a parameter and removes duplicate rows
    based on all columns. It returns the DataFrame without duplicates.

    Parameters:
    df_params (DataFrame): The input DataFrame from which duplicates will be removed.

    Returns:
    DataFrame: The DataFrame after removing duplicate rows.
    """
    # Remove duplicates across all columns
    df_params.drop_duplicates(inplace=True)
    return df_params


def clear_missing_data(df):

    if df.isnull().all().sum() != 0:
        df.dropna(axis=1, how='all', inplace=True)
    # If there are no empty rows, no operation is performed
    else:
        print("No empty rows")

def delete_empty_percent(df, threshold):
    """
    Identifies columns with more than threshold% missing values

    Args:
        df (pandas.DataFrame): DataFrame containing missing values

    Returns:
        pandas.Series, int: Result with more than 80% missing values
    """
    percent_missing = df.isnull().sum() * 100 / len(df)
    percent_missing.sort_values(ascending=False, inplace=True)

    threshold_view = 2
    filtered = percent_missing[percent_missing.values > threshold_view]

  #  threshold = 80
    columns_to_drop = percent_missing[percent_missing.values > threshold].index
    count_columns_to_drop = len(columns_to_drop)

    columns_to_drop_details = percent_missing[percent_missing.values > threshold]

    return columns_to_drop_details, threshold

def non_useful_columns(df_params):
    """
    This function takes a DataFrame as a parameter, removes specific non-useful columns,
    and returns a cleaned DataFrame.

    Parameters:
    df_params (DataFrame): The input DataFrame from which unnecessary columns will be removed.

    Returns:
    DataFrame: The DataFrame after removing the specified columns.
    """
    colonnes_a_supprimer = [
        "url", "created_t", "created_datetime", "last_modified_t", "last_modified_datetime",
        "last_modified_by", "last_updated_t", "brands_tags", "last_updated_datetime",
        "countries_tags", "countries_en", "states_tags", "states_en", "image_url",
        "image_small_url", "image_nutrition_url", "image_nutrition_small_url"
    ]

    df_params = df_params.drop(columns=colonnes_a_supprimer)
    return df_params

