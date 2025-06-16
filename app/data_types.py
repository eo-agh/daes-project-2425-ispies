"""A module defining data types for the application, including a ModelBank class."""

import copy
from dataclasses import dataclass
from typing import Literal

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from statsforecast.models import AutoARIMA, AutoETS, HistoricAverage

from app import constants

Metric = Literal[constants.MAPE, constants.MAE, constants.RMSE]

METRIC_BANK = {
    constants.MAPE: mean_absolute_percentage_error,
    constants.MAE: mean_absolute_error,
    constants.RMSE: root_mean_squared_error,
}


@dataclass
class ModelBank:
    """
    Represents a bank of models with a name and a list of model names.
    """

    models = {
        constants.MODEL_AUTO_ARIMA_365: AutoARIMA(season_length=365),
        constants.MODEL_AUTO_ETS_365: AutoETS(season_length=365),
        constants.MODEL_HISTORIC_AVERAGE: HistoricAverage(),
    }

    def get(self, model_name: str):
        """
        Returns the model instance corresponding to the given model name.
        """
        if model_name not in self.models.keys():
            raise ValueError(f"Model '{model_name}' not found in ModelBank.")

        return copy.deepcopy(self.models.get(model_name))
