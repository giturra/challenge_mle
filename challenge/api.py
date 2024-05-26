from typing import List

import fastapi
import pandas as pd
from pydantic import BaseModel

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


class Flights(BaseModel):
    flights: List[Flight]


app = fastapi.FastAPI()
delay_model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


def preprocess_flights(flights: Flights):
    data_df = pd.DataFrame(flights.dict()["flights"])
    data_dict = delay_model.prepocess_dataset(pd.DataFrame(data_df)).to_dict()

    for key in TOP_10_FEATURES:
        if key not in data_dict:
            data_dict[key] = [0]

    remove_columns = [key for key in data_dict if key not in TOP_10_FEATURES]

    for key in remove_columns:
        data_dict.pop(key)

    return pd.DataFrame(data_dict)


@app.post("/predict", status_code=200)
async def post_predict(flights: Flights) -> dict:
    req_data_df = preprocess_flights(flights)
    return {"predict": delay_model.predict(req_data_df)}
