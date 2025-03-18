"""
Fugit nction for the task 0, will show numerical and categorical columns
Will also reduce the size of floats to increase performances
Args : df -> the dataframe to be modified
Return : df -> the modified dataframe
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



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
    """
    Removes empty columns from the DataFrame

    Args:
        df (pandas.DataFrame): DataFrame containing missing values

    Returns:
        pandas.DataFrame: DataFrame without empty columns
    """
    missing_values = df.isnull().sum() / len(df) * 100
  

    empty_columns_count = missing_values[missing_values == 100].count()
    print(f"Nombre de colonnes vides (100% de valeurs manquantes) : {empty_columns_count}")

    df = df.dropna(axis=1, how='all')

    return df

import pandas as pd

def clear_missing_line(df):
    """
    Removes empty rows from the DataFrame

    Args:
        df (pandas.DataFrame): DataFrame containing missing values

    Returns:
        pandas.DataFrame: DataFrame without empty rows
    """
    missing_values = df.isnull().sum(axis=1) / df.shape[1] * 100

    empty_rows_count = missing_values[missing_values == 100].count()
    print(f"Nombre de lignes vides (100% de valeurs manquantes) : {empty_rows_count}")

    df = df.dropna(axis=0, how='all')

    return df


def delete_columnns_treshold(df, threshold=80):
      
    percent_missing = df.isnull().sum() * 100 / len(df) # Calculate how empty each columns are
    percent_missing.sort_values(ascending=False, inplace=True) # Order them

    filtered = percent_missing[percent_missing.values > threshold] # Keep only columns where empty% > Threshold
    columns_to_drop = percent_missing[
        percent_missing.values > threshold].index # Keep only the column name, we drop the percentages for now

    df.drop(columns_to_drop, axis='columns', inplace=True) # Drop the columns
    print(f"Les colonnes supprimées sont : {columns_to_drop}")
    return df




def non_useful_columns(df,df_params=None):
    """
    This function takes a DataFrame as a parameter, removes specific non-useful columns,
    and returns a cleaned DataFrame. If no DataFrame is provided, it returns a default list
    of columns to be removed.

    Parameters:
    df_params (DataFrame, optional): The input DataFrame from which unnecessary columns will be removed.

    Returns:
    DataFrame or list: The DataFrame after removing the specified columns, or a list of default columns.
    """
    colonnes_a_supprimer = [
        "url", "created_t", "created_datetime", "last_modified_t", "last_modified_datetime",
        "last_modified_by", "last_updated_t", "brands_tags", "last_updated_datetime",
        "countries_tags", "countries_en", "states_tags", "states_en", "image_url",
        "image_small_url", "image_nutrition_url", "image_nutrition_small_url"
    ]

    if df_params is None:
        df_params = colonnes_a_supprimer

    print(f"Les colonnes supprimées sont : {df_params}")

    return df.drop(columns=df_params)

def visualize(df, columns_to_drop, threshold):
    """
    Displays a graph of columns with too many missing values

    Args:
        df (pandas.DataFrame): 
        columns_to_drop_details (pandas.Series): 
        threshold (int): 

    Returns:
        pandas.DataFrame: The original DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=columns_to_drop.index, y=columns_to_drop.values)
    plt.xticks(rotation=90)
    plt.xlabel('Colonnes')
    plt.ylabel('% de valeurs manquantes')
    plt.title(f"Colonnes avec plus de {threshold}% de valeurs manquantes")
    plt.tight_layout()
    plt.show()

    return df
#