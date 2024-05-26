import fastapi
from pydantic import BaseModel


class Flights(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.get("/predict", status_code=200)
async def post_predict(flights: Flights) -> dict:
    return flights.model_dump()
