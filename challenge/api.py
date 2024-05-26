from typing import List

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
    def valid_month_number(cls, month):
        month_numbers = list(range(1, 13))
        if month not in month_numbers:
            raise ValueError(
                f"MES must be a number between: {month_numbers}, but you give {month}."
            )
        return month

    @validator("OPERA")
    def valid_flight_airline(cls, airline):
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
    def valid_flight_type(cls, flight_type):
        validate_flight_type = ["I", "N"]
        if flight_type not in validate_flight_type:
            raise ValueError(
                f"TIPOVUELO must be an flight type between: {validate_flight_type}, but"
                f" you give {flight_type}."
            )
        return flight_type


class Flights(BaseModel):
    flights: List[Flight]


app = FastAPI()
delay_model = DelayModel()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


def preprocess_flights(flights: Flights):
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
async def post_predict(flights: Flights) -> dict:
    req_data_df = preprocess_flights(flights)
    return {"predict": delay_model.predict(req_data_df)}
