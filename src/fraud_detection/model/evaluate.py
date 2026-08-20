from sklearn.metrics import f1_score, classification_report, average_precision_score, precision_recall_curve
from src.fraud_detection.utils import logger
import numpy as np
import logging

logger = logging.getLogger(__name__)


def evaluate(y, y_pred, message):
    logger.info(f"starting evaluate {message}")
    f1 = f1_score(y, y_pred)
    average_precision = average_precision_score(y, y_pred)
    logger.info(f"f1 = {f1} , Avg_precision = {average_precision} for {message}")
    print("*" * 50)
    print(message)
    print('*' * 50)
    print(f"f1 score = {f1}")
    print(f"Avg_precision = {average_precision}")
    print()
    print("classification report")
    print(classification_report(y, y_pred))


def calc_best_thershold(y, y_pred):
    """
    Find the probability threshold that maximizes the F1 score.

    Parameters
    ----------
    y : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities (scores) for the positive class.

    Returns
    -------
    best_threshold : float
        The threshold that yields the highest F1 score.
    best_f1 : float
        The F1 score achieved at best_threshold.
    """
    logger.info("starting calc_best_thershold")

    precision, recall, thresholds = precision_recall_curve(y, y_pred)

    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    # so we align precision/recall with thresholds by dropping the last point
    precision = precision[:-1]
    recall = recall[:-1]

    # avoid division by zero
    denom = precision + recall + 1e-12
    f1_scores = 2 * precision * recall / denom

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    logger.info(f"best_threshold = {best_threshold} , best_f1 = {best_f1}")
    print(f"Best threshold = {best_threshold}")
    print(f"Best F1 score at this threshold = {best_f1}")

    return best_threshold, best_f1