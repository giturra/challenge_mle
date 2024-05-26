from typing import List

import fastapi
import pandas as pd
from pydantic import BaseModel

from challenge.model import DelayModel


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class Flights(BaseModel):
    flights: List[Flight]


app = fastapi.FastAPI()
model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


def preprocess_flights(flights: Flights):
    raw_data = flights.dict()["flights"]
    return raw_data


@app.post("/predict", status_code=200)
async def post_predict(flights: Flights) -> dict:
    raw_data_list = flights.dict()["flights"]
    raw_data_df = model.prepocess_dataset(pd.DataFrame(raw_data_list))
    print(raw_data_df)
    return flights.dict()["flights"]
