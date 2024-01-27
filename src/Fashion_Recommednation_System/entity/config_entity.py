from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size:list
    params_include_top:bool
    params_weights:str

@dataclass(frozen=True)
class TrainingConfig:
    root_dir:Path
    training_data:Path
    updated_base_model_path:Path
    saved_features:Path
    image_paths:Path

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model:Path
    training_data:Path
    extract_feature:Path
    image_paths:Path
    mlflow_uri:str
     




