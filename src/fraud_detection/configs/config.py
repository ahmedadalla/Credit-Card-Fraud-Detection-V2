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
        "alpha": 0.001,
        "max_iter": 10000,
        "early_stopping": True,
        "n_iter_no_change": 15,
        "random_state": 42
    })

    knn: Dict[str, Any] = field(default_factory=lambda: {
        "n_neighbors": 3,
        "weights": "distance",
        "algorithm": "auto",
        "n_jobs": -1
    })

@dataclass
class GridSearchConfig:
    logistic_regression: Dict[str, Any] = field(default_factory=lambda: {
            "model__class_weight": [{0: 1, 1: 2},{0: 1, 1:4 },{0: 1, 1: 1},{0: 1, 1: 6},{0: 1, 1: 10},{0: 1, 1: 15}],
            "model__max_iter": [10000],
            "model__random_state": [42]
        })
        
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        "model__n_estimators": [50,100,300,500],
        "model__max_depth": [3,7,12],
        "model__min_samples_split": [5,10],
        "model__min_samples_leaf": [2,6],
        "model__class_weight": [{0: 1, 1: 2},{0: 1, 1: 1},{0: 1, 1: 6}],
        "model__random_state": [42],
        "model__n_jobs": [-1]
    })

    mlp: Dict[str, Any] = field(default_factory=lambda: {
        "model__hidden_layer_sizes": [(60,),(60,30),(128, 64)],
        "model__activation": ["relu"],
        "model__solver": ["adam"],
        "model__alpha": [0.1,0.01,0.001,0.00001,0.0001],
        "model__max_iter": [10000],
        "model__early_stopping": [True],
        "model__n_iter_no_change": [15],
        "model__random_state": [42]
    })

    knn: Dict[str, Any] = field(default_factory=lambda: {
        "model__n_neighbors": [15,10,5,3],
        "model__weights": ["distance",None],
        "model__algorithm": ["auto"],
        "model__n_jobs": [-1]
    })

@dataclass
class TrainingConfig:
    """Configuration for training process."""

    evaluation_metric: str = "f1"


@dataclass
class Config:
    """Main configuration class."""
    grid:GridSearchConfig=field(default_factory=GridSearchConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Create a global config instance
config = Config()


from pydantic_settings import BaseSettings, SettingsConfigDict

# api config
class Settings(BaseSettings):
    MODEL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()