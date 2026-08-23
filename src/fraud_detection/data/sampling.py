from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.fraud_detection.configs import Paths


def create_sampler(
    strategy="none",
    under_factor=1.0,
    over_factor=0.2,
    random_state=42
):
    if strategy == "none":
        return None

    if strategy == "under":
        return RandomUnderSampler(
            sampling_strategy=under_factor,
            random_state=random_state
        )

    if strategy == "over":
        return SMOTE(
            sampling_strategy=over_factor,
            random_state=random_state
        )

    if strategy == "over_under":
        return create_over_under_sampler(
            under_factor=under_factor,
            over_factor=over_factor,
            random_state=random_state
        )

    raise ValueError(
        f"Unknown sampling strategy: {strategy}"
    )


def create_over_under_sampler(
    under_factor=1.0,
    over_factor=0.5,
    random_state=42
):
    return ImbPipeline([
        (
            "over",
            SMOTE(
                sampling_strategy=over_factor,
                random_state=random_state
            )
        ),
        (
            "under",
            RandomUnderSampler(
                sampling_strategy=under_factor,
                random_state=random_state
            )
        )
    ])