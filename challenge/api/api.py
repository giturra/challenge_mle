from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from challenge.api.models import TOP_10_FEATURES, Flights
from challenge.model import DelayModel

app = FastAPI()
delay_model = DelayModel()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle validation errors and return a 400 Bad Request response.

    Args:
        request (Request): The incoming request.
        exc (RequestValidationError): The validation error.

    Returns:
        JSONResponse: A JSON response with the error details.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


@app.get("/health", status_code=200)
async def get_health() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        dict: A dictionary indicating the status of the service.
    """
    return {"status": "OK"}


def preprocess_flights(flights: Flights) -> pd.DataFrame:
    """
    Preprocess the flight data for prediction.

    Args:
        flights (Flights): The flight data to preprocess.

    Returns:
        pd.DataFrame: The preprocessed flight data.
    """
    data_df = pd.DataFrame(flights.dict()["flights"])
    data_dict = delay_model.prepocess_dataset(pd.DataFrame(data_df)).to_dict()
    for key in TOP_10_FEATURES:
        if key not in data_dict:
            data_dict[key] = [0]
        else:
            data_dict[key] = [1]

    remove_columns = [key for key in data_dict if key not in TOP_10_FEATURES]
    for key in remove_columns:
        data_dict.pop(key)
    ordered_keys = sorted(data_dict.keys(), key=lambda x: TOP_10_FEATURES.index(x))

    ordered_dict = {key: data_dict[key] for key in ordered_keys}
    return pd.DataFrame(ordered_dict)


@app.post("/predict", status_code=200)
async def post_predict(flights: Flights) -> Dict[str, Any]:
    """
    Predict flight delays based on the provided flight data.

    Args:
        flights (Flights): The flight data for prediction.

    Returns:
        dict: A dictionary containing the prediction results.
    """
    req_data_df = preprocess_flights(flights)
    return {"predict": delay_model.predict(req_data_df)}
