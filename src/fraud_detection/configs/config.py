from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DataConfig:
    """Configuration for data processing."""

    target: str = "Class"


@dataclass
class SamplingConfig:
    """Configuration for data sampling."""
    under_factor: float = 0.02
    over_factor: float = 0.05
    random_state: int = 42
    strategy: str = "over_under"
    # Options: "none", "under", "over", "over_under"


@dataclass
class ModelConfig:
    """Configuration for model training."""

    logistic_regression: Dict[str, Any] = field(default_factory=lambda: {
        "solver": "lbfgs",
        "class_weight": {0: 1, 1: 2},
        "max_iter": 10000,
        "random_state": 42
    })
    
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1
    })

    mlp: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate": "adaptive",
        "max_iter": 1000,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "random_state": 42
    })

    knn: Dict[str, Any] = field(default_factory=lambda: {
        "n_neighbors": 15,
        "weights": "distance",
        "algorithm": "auto",
        "n_jobs": -1
    })


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    evaluation_metric: str = "f1"


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Create a global config instance
config = Config()