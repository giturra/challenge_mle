from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.config import Config
from challenge.model import DelayModel

config = Config(config_path="challenge/configs/api_config.yaml")

TOP_10_FEATURES = config.get("top_10_features", [])
AIRLINES = config.get("airlines", [])
MONTHS = config.get("months", [])
FLIGHT_TYPES = config.get("flight_type", [])


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def valid_month_number(cls, month: int) -> int:
        """Validate that the month is between 1 and 12."""
        if month not in MONTHS:
            raise ValueError(
                f"MES must be a number between: {MONTHS}, but you give {month}."
            )
        return month

    @validator("OPERA")
    def valid_flight_airline(cls, airline: str) -> str:
        """Validate that the airline is in the list of valid airlines."""
        if airline not in AIRLINES:
            raise ValueError(
                f"OPERA must be an airline between: {AIRLINES}, but you give "
                f"{airline}."
            )
        return airline

    @validator("TIPOVUELO")
    def valid_flight_type(cls, flight_type: str) -> str:
        """Validate that the flight type is either 'I' or 'N'."""
        if flight_type not in FLIGHT_TYPES:
            raise ValueError(
                f"TIPOVUELO must be a flight type between: {FLIGHT_TYPES}, but"
                f" you give {flight_type}."
            )
        return flight_type


class Flights(BaseModel):
    flights: List[Flight]


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
    print(data_dict.keys())
    for key in TOP_10_FEATURES:
        if key not in data_dict:
            data_dict[key] = [0]
        else:
            data_dict[key] = [1]

    remove_columns = [key for key in data_dict if key not in TOP_10_FEATURES]
    print(remove_columns)
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
