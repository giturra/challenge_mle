from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_min_diff
from xgboost import XGBClassifier


class DelayModel:
    def __init__(self) -> None:
        self.__top_10_features = [
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

        self.random_state = 1
        self.learning_rate = 0.01

        self._model = None  # Model should be saved in this attribute.

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        data = self.__prepocess_dataset(data=data, threshold_in_minutes=15)
        if target_column:
            features = data[self.__top_10_features]
            target = data[target_column]
            return features, target

        return data

    def __prepocess_dataset(
        self, data: pd.DataFrame, threshold_in_minutes: int
    ) -> pd.DataFrame:
        data["min_diff"] = data.apply(get_min_diff, axis=1)
        data["delay"] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)
        return data

    def __scale_pos_weight(self, target: pd.DataFrame) -> None:
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1
        return scale

    def __split_dataset(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        test_size: float = 0.33,
        random_state=123,
        shuffle: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        return x_train, x_test, y_train, y_test

    def get_model(self) -> XGBClassifier:
        return self._model

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        scale: float = self.__scale_pos_weight(target=target)
        self._model = XGBClassifier(
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale,
        )
        x_train, _, y_train, _ = self.__split_dataset(features=features, target=target)
        self.model.fit(x_train, y_train)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return
