from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    root_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    trained_model_path: Path
    training_data: Path
    params_epochs: int
    params_n_classes: int
    params_n_freeze_epochs: int
    params_patience: int
    params_model_name: str