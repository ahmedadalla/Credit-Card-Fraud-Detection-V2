from src.fraud_detection.data.sampling import create_sampler
from src.fraud_detection.model.model_factory import create_model
from src.fraud_detection.model.utils import set_model_params,save_model,get_model_params
from src.fraud_detection.model.evaluate import evaluate,calc_best_thershold
from src.fraud_detection.pipline.pipline import make_pipeline
from src.fraud_detection.data.loader import Load_Processed,split_to_target_and_features
from src.fraud_detection.utils import logger
from src.fraud_detection.configs import Paths
from src.fraud_detection.configs.config import config
import argparse
import logging

logger=logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train a machine learning model")

parser.add_argument("--model", type=str,  help="model to choose from"
                                                            "lg > logistic"
                                                            "rf > randome forest"
                                                            "nn > nural_netowrks" \
                                                            "knn > KNeighborsClassifier"
                                                            )
parser.add_argument("--sampling", type=str,  help="none > nothing"
                                                                   "under > under_sampling"
                                                                   "over > over_sampling"
                                                                   "over_under > .over + under",default="none")

if __name__ == "__main__":
    args = parser.parse_args()

    logger.info("loading data")
    train_df=Load_Processed("train_processed.csv")
    val_df=Load_Processed("train_val_processed.csv")

    logger.info("splitting to target and features")
    x_train,y_train=split_to_target_and_features(train_df,config.data.target)
    x_val,y_val=split_to_target_and_features(val_df,config.data.target)

    logger.info("getting model and set its params")
    model=create_model(args.model)
    model_params=get_model_params(args.model)
    set_model_params(model,model_params)

    logger.info("create the sampler")
    sampler = create_sampler(strategy=args.sampling, over_factor=config.sampling.over_factor, under_factor=config.sampling.under_factor)
    
    logger.info("creating the pipline")
    pipeline=make_pipeline(sampler=sampler,model=model)

    logger.info("fitting the pipeline")
    pipeline.fit(x_train,y_train)

    logger.info("evaluation")
    y_train_pred=pipeline.predict(x_train)
    evaluate(y_train,y_train_pred,"training")

    y_val_pred=pipeline.predict(x_val)
    evaluate(y_val,y_val_pred,"validation")
    calc_best_thershold(y_val,y_val_pred)
    