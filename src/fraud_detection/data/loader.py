import pandas as pd
from src.fraud_detection.configs import Paths

def Load_Raw(file_name:str):
    return pd.read_csv(Paths.RAW_DATA+file_name)

def Load_Processed(file_name:str):
    return pd.read_csv(Paths.PROCESSED+file_name)

def Save_Raw(df:pd.DataFrame,file_name:str):
    df.to_csv(Paths.RAW_DATA+file_name,index=False)

def Save_Processed(df:pd.DataFrame,file_name:str):
    df.to_csv(Paths.PROCESSED+file_name,index=False)

def split_to_target_and_features(
    df: pd.DataFrame,
    target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_column : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Feature columns.
    y : pd.Series
        Target column.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y
    