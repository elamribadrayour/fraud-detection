"""fraud detection api lifespan."""

import os
from contextlib import asynccontextmanager

import xgboost
from fastapi import FastAPI

from helpers.env import cache_path


model: xgboost.Booster


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model()
    yield
    del model


def get_model() -> xgboost.Booster:
    model_path = os.path.join(cache_path, "model/model.ubj")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = xgboost.Booster()
    model.load_model(model_path)
    return model
