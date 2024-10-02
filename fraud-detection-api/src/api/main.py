from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from helpers import lifespan
from helpers.inputs import Input

app = FastAPI(
    lifespan=lifespan.lifespan,
    title="Fraud Detection API",
)


@app.post("/predict/")
async def predict(input_: Input) -> ORJSONResponse:
    prediction_probs = lifespan.model.predict(input_.to_dmatrix())
    prediction = int(prediction_probs[0] > 0.5)
    return ORJSONResponse(
        content={"prediction": prediction, "probability": float(prediction_probs[0])}
    )
