from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel

TOP_10_FEATURES = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def valid_month_number(cls, month: int) -> int:
        """Validate that the month is between 1 and 12."""
        month_numbers = list(range(1, 13))
        if month not in month_numbers:
            raise ValueError(
                f"MES must be a number between: {month_numbers}, but you give {month}."
            )
        return month

    @validator("OPERA")
    def valid_flight_airline(cls, airline: str) -> str:
        """Validate that the airline is in the list of valid airlines."""
        validate_airlines = [
            "Aerolineas Argentinas",
            "Aeromexico",
            "Air Canada",
            "Air France",
            "Alitalia",
            "American Airlines",
            "Austral",
            "Avianca",
            "British Airways",
            "Copa Air",
            "Delta Air",
            "Gol Trans",
            "Grupo LATAM",
            "Iberia",
            "JetSmart SPA",
            "K.L.M.",
            "Lacsa",
            "Latin American Wings",
            "Oceanair Linhas Aereas",
            "Plus Ultra Lineas Aereas",
            "Qantas Airways",
            "Sky Airline",
            "United Airlines",
        ]
        if airline not in validate_airlines:
            raise ValueError(
                f"OPERA must be an airline between: {validate_airlines}, but you give "
                f"{airline}."
            )
        return airline

    @validator("TIPOVUELO")
    def valid_flight_type(cls, flight_type: str) -> str:
        """Validate that the flight type is either 'I' or 'N'."""
        validate_flight_type = ["I", "N"]
        if flight_type not in validate_flight_type:
            raise ValueError(
                f"TIPOVUELO must be a flight type between: {validate_flight_type}, but"
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
