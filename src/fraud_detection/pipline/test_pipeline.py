from src.fraud_detection.model.utils import load_model 
from src.fraud_detection.model.evaluate import evaluate,calc_best_thershold
from src.fraud_detection.data.loader import Load_Processed,split_to_target_and_features
from src.fraud_detection.configs.config import config
import argparse
from src.fraud_detection.utils import logger
import logging

logger=logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train a machine learning model")

parser.add_argument("--model", type=str,  help="model to choose from"
                                                            "lg > logistic"
                                                            "rf > randome forest"
                                                            "nn > nural_netowrks" \
                                                            "knn > KNeighborsClassifier"
                                                            )

if __name__ == "__main__":
    args = parser.parse_args()

    logger.info("loading data")
    test_df=Load_Processed("test_processed.csv")

    logger.info("splitting to target and features")
    x_test,y_test=split_to_target_and_features(test_df,config.data.target)

    logger.info ("loading the model")
    if args.model:
        model=load_model(args.model+"_pip.pkl")

    logger.info("evaluation")
    y_pred=model.predict(x_test)
    evaluate(y_test,y_pred,"test")


    y_porba=model.predict_proba(x_test)[:, 1]
    calc_best_thershold(y_test,y_porba)