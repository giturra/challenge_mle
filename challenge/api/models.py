from typing import List

from pydantic import BaseModel, validator

from challenge.config import Config

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
