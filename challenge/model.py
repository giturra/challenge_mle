import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import Config
from .utils import get_min_diff


class DelayModel:
    """
    A model for predicting flight delays based on various features.

    Attributes:
        config (Config): Configuration object to load model parameters.
        model_name (str): Name of the model.
        model_version (str): Version of the model.
        random_state (int): Random state for reproducibility.
        learning_rate (float): Learning rate for the XGBoost model.
        __top_10_features (List[str]): List of top 10 features used for prediction.
        __raw_data_columns (List[str]): List of raw data columns.
        _model (XGBClassifier): The XGBoost model instance.
        threshold_in_minutes (int): Threshold in minutes to determine delay.
        trained_model_path (str): Path to save the trained model.
    """

    def __init__(self) -> None:
        """
        Initialize the DelayModel class.
        """
        self.config = Config(config_path="challenge/configs/model_config.yaml")

        self.model_name = self.config.get("model_name", "xgboost_classifier")
        self.model_version = self.config.get("model_version", "1.0")

        self.random_state = self.config.get("random_state", 1)
        self.learning_rate = self.config.get("learning_rate", 0.01)

        self.__top_10_features = self.config.get("top_10_features", default=[])
        self.__raw_data_columns = self.config.get("raw_data_columns", default=[])

        self._model = None

        self.threshold_in_minutes = self.config.get("threshold_in_minutes", 15)

        self.trained_model_path = self.__create_path_for_save_trained_models()

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

        features = self.prepocess_dataset(data=data)
        if target_column:
            data["min_diff"] = data.apply(get_min_diff, axis=1)
            data[target_column] = np.where(
                data["min_diff"] > self.threshold_in_minutes, 1, 0
            )
            target = data[[target_column]]
            return features[self.__top_10_features], target

        return features[self.__top_10_features]

    def __create_path_for_save_trained_models(self) -> str:
        """
        Create a directory to save trained models.

        Returns:
            str: Path to the directory where trained models are saved.
        """
        dir_path = f"{os.getcwd()}/trained_models"
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        return dir_path

    def prepocess_dataset(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the dataset by calculating the delay and creating features.

        Args:
            data (pd.DataFrame): Raw data.
            threshold_in_minutes (int): Threshold in minutes to determine delay.
            target_column (str): Name of the target column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed data and features.
        """

        features = pd.concat(
            [
                pd.get_dummies(data[column], prefix=column)
                for column in self.__raw_data_columns
            ],
            axis=1,
        )
        return features

    def __scale_pos_weight(self, target: pd.DataFrame) -> float:
        """
        Calculate the scale_pos_weight parameter for the XGBoost model.

        Args:
            target (pd.DataFrame): Target data.

        Returns:
            float: Scale position weight.
        """
        n_y0 = len(target[target["delay"] == 0])
        n_y1 = len(target[target["delay"] == 1])
        scale = n_y0 / n_y1
        return scale

    def __split_dataset(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        test_size: float = 0.33,
        random_state=123,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets.

        Args:
            features (pd.DataFrame): Features data.
            target (pd.DataFrame): Target data.
            test_size (float, optional): Proportion of the dataset to include in the
                test split. Defaults to 0.33.
            random_state (int, optional): Random state for reproducibility. Defaults
            to 123.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training and
                testing features and targets.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        return x_train, x_test, y_train, y_test

    def __save_model(self) -> None:
        """
        Save the trained model to a file.
        """
        self._model.save_model(
            f"{self.trained_model_path}/{self.model_name}{self.model_version}.json"
        )

    def __load_model(self) -> XGBClassifier:
        """
        Load the trained model from a file.

        Returns:
            XGBClassifier: Loaded XGBoost model.
        """
        loaded_model = XGBClassifier()
        loaded_model.load_model(
            f"{self.trained_model_path}/{self.model_name}{self.model_version}.json"
        )
        return loaded_model

    def get_model(self) -> XGBClassifier:
        """
        Get the trained model. Load it if it is not already loaded.

        Returns:
            XGBClassifier: Trained XGBoost model.
        """
        if self._model is None:
            self._model = self.__load_model()
        return self._model

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, _, y_train, _ = self.__split_dataset(features=features, target=target)
        scale: float = self.__scale_pos_weight(target=y_train)
        self._model = XGBClassifier(
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale,
        )
        self._model.fit(x_train, y_train)
        self.__save_model()

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        predicted_targets = self.get_model().predict(features)
        return [1 if y_pred > 0.5 else 0 for y_pred in predicted_targets]
