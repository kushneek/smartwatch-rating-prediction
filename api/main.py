

import sys
from pathlib import Path

# Allow `from smartwatch_ml... import` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from smartwatch_ml.predict import predict as run_prediction, get_metadata
from .schemas import SmartwatchInput, PredictionResponse, ModelInfoResponse

app = FastAPI(
    title="Smartwatch Rating Prediction API",
    description="Predicts customer rating (1-5) for a smartwatch based on its specs.",
    version="1.0.0",
)

# Local dev: allow any origin so a frontend running on a different port/host
# (or a Claude artifact running in your browser) can call this API.
# Tighten this to specific origins before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    metadata = get_metadata()
    return ModelInfoResponse(**metadata)

@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


@app.post("/predict", response_model=PredictionResponse)
def predict_rating(payload: SmartwatchInput):
    raw = payload.model_dump(by_alias=True)
    try:
        result = run_prediction(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionResponse(**result)