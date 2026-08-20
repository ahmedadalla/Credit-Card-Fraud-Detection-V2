from src.fraud_detection.data import loader
from src.fraud_detection.utils import logger
from src.fraud_detection.configs import Paths
import logging

logger=logging.getLogger(__name__)

if __name__=="__main__":
    logger.info("Starting data cleaning")

    logger.info(f"Loading data")
    train_df = loader.Load_Raw("test.csv")

    null_count = train_df.isna().sum().sum()
    logger.info(f"Total null values: {null_count}")
    print(f'nulls count : {null_count}')

    duplicates = train_df.duplicated().sum()
    logger.info(f"Total duplicates values: {duplicates}")
    print(f"duplicates count : {duplicates}")

    logger.info("removing duplicates")
    train_df=train_df.drop_duplicates()

    logger.info("saving the data")
    loader.Save_Processed(train_df,"test_processed.csv")



