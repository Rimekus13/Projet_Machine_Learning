import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from sklearn.preprocessing import OneHotEncoder



def imputation_of_categorical_val(df_params):
    """
    This function imputes missing values in categorical variables using the most frequent value.

    Parameters:
    df_params (DataFrame): The input DataFrame containing categorical variables with missing values.

    Returns:
    DataFrame: The DataFrame after imputing missing categorical values.
    """
    # Select categorical columns (type object or category)
    categorical_cols = df_params.select_dtypes(include=["object", "category"]).columns
    # Impute missing values with the most frequent value in each column
    imputer = SimpleImputer(strategy="most_frequent")
    df_params[categorical_cols] = imputer.fit_transform(df_params[categorical_cols])
    return df_params


def imputation_of_numerical_val(df_params):
    """
    This function imputes missing values in numerical variables using the k-nearest neighbors (KNN) imputation method.

    Parameters:
    df_params (DataFrame): The input DataFrame containing numerical variables with missing values.

    Returns:
    DataFrame: The DataFrame after imputing missing numerical values.
    """
    # Select numerical columns (type float or int)
    numeric_features = df_params.select_dtypes(include=['float', 'int'])
    # Impute missing values using K-Nearest Neighbors
    imputer = KNNImputer(missing_values=np.nan)
    imputed_values = imputer.fit_transform(numeric_features)
    # Replace the original numerical columns with the imputed values
    df_params.loc[:, numeric_features.columns] = imputed_values
    return df_params

def onehotencoder(df):

    # Init, the argument 'sparse_output=False' ensure we get a dense matrix
    encoder = OneHotEncoder(sparse_output=False)

    # Apply onehotencoder to all categorical columns
    encoded_columns = encoder.fit_transform(df.select_dtypes(include=['object']))

    # Convert the encoded columns to a DataFrame and assign meaningful column names
    encoded_df = pd.DataFrame(encoded_columns,columns=encoder.get_feature_names_out(df.select_dtypes(include=['object']).columns))

    # Concatenate the encoded columns with the original DataFrame and drop the original categorical columns
    df = pd.concat([df, encoded_df], axis=1).drop(columns=df.select_dtypes(include=['object']).columns)

    return df