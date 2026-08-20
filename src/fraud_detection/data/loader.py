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
    