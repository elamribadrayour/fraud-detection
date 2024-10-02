"""paths."""

import os


def get_data_path(cache_path: str) -> str:
    data_path = os.path.join(cache_path, "data")
    os.makedirs(data_path, exist_ok=True)
    return data_path


def get_train_data_path(cache_path: str) -> str:
    file_path = os.path.join(cache_path, "data/train.parquet")
    if os.path.exists(file_path) is False:
        raise IOError(f"file doesn't exist {file_path}")
    return file_path


def get_test_data_path(cache_path: str) -> str:
    file_path = os.path.join(cache_path, "data/test.parquet")
    if os.path.exists(file_path) is False:
        raise IOError(f"file doesn't exist {file_path}")
    return file_path


def get_model_path(cache_path: str) -> str:
    data_path = os.path.join(cache_path, "model")
    os.makedirs(data_path, exist_ok=True)
    return data_path
