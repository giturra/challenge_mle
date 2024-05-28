from typing import List

from pydantic import BaseModel, validator

from challenge.config import Config

config = Config(config_path="challenge/configs/api_config.yaml")

TOP_10_FEATURES = config.get("top_10_features", [])
AIRLINES = config.get("airlines", [])
MONTHS = config.get("months", [])
FLIGHT_TYPES = config.get("flight_type", [])


class Flight(BaseModel):
    """
    A model representing a flight with its airline, flight type, and month.

    Attributes:
        OPERA (str): The airline of the flight.
        TIPOVUELO (str): The type of the flight ('I' or 'N').
        MES (int): The month of the flight (1-12).
    """

    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def valid_month_number(cls, month: int) -> int:
        """
        Validate that the month is between 1 and 12.

        Args:
            month (int): The month number.

        Raises:
            ValueError: If the month is not between 1 and 12.

        Returns:
            int: The validated month number.
        """
        if month not in MONTHS:
            raise ValueError(
                f"MES must be a number between: {MONTHS}, but you give {month}."
            )
        return month

    @validator("OPERA")
    def valid_flight_airline(cls, airline: str) -> str:
        """
        Validate that the airline is in the list of valid airlines.

        Args:
            airline (str): The airline name.

        Raises:
            ValueError: If the airline is not in the list of valid airlines.

        Returns:
            str: The validated airline name.
        """
        if airline not in AIRLINES:
            raise ValueError(
                f"OPERA must be an airline between: {AIRLINES}, but you give "
                f"{airline}."
            )
        return airline

    @validator("TIPOVUELO")
    def valid_flight_type(cls, flight_type: str) -> str:
        """
        Validate that the flight type is either 'I' or 'N'.

        Args:
            flight_type (str): The flight type.

        Raises:
            ValueError: If the flight type is not 'I' or 'N'.

        Returns:
            str: The validated flight type.
        """
        if flight_type not in FLIGHT_TYPES:
            raise ValueError(
                f"TIPOVUELO must be a flight type between: {FLIGHT_TYPES}, but"
                f" you give {flight_type}."
            )
        return flight_type


class Flights(BaseModel):
    """
    A model representing a list of flights.

    Attributes:
        flights (List[Flight]): A list of Flight objects.
    """

    flights: List[Flight]
