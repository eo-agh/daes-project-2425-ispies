"""Forecaster Class"""

from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast

from app import constants
from app.data_types import ModelBank


class Forecaster:
    """
    A class for forecasting time series data using
    the best model determined by cross-validation.
    """

    def __init__(
        self,
        cv_results: pd.DataFrame,
        freq: str,
        forecast_horizon: int = 7,
        level: list[int] = [68],
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        """
        Initializes the Forecaster with cross-validation results.

        Parameters
        ----------
        cv_results : pd.DataFrame
            DataFrame containing the cross-validation results with columns:
            - unique_id: Unique identifier for the time series.
            - start_date: First cutoff date for predictions with the best model.
            - best_model: The name of the best model determined by cross-validation.
        freq : str
            Frequency of the time series data (e.g., 'D' for daily, 'H' for hourly).
        forecast_horizon : int, optional
            The number of periods to forecast ahead (default is 7).
        level : list[int], optional
            Confidence levels for prediction intervals (default is [68]).
        n_jobs : int, optional
            Level of parallelism for forecasting (default is 1).
        verbose : bool, optional
            Whether to print verbose output (default is True).
        """
        if len(level) > 1:
            raise NotImplementedError(
                "Multiple confidence levels are not supported yet."
            )

        if len(level) == 0:
            raise NotImplementedError("At least one confidence level must be provided.")

        self.cv_results = cv_results
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.level = level
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> Forecaster:
        """
        Fits the Forecaster to the provided time series data.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the time series data with a 'unique_id' column
            'ds' column for timestamps and other relevant columns for forecasting.

        Returns
        -------
        Forecaster
            The fitted Forecaster instance.
        """
        cv_results = self.cv_results.query(
            f"`{constants.START_DATE}` <= '{X[constants.TIMESTAMP_COLUMN].max()}'"
        )
        self.cv_results = cv_results.query(
            f"`{constants.START_DATE}` == '{cv_results[constants.START_DATE].max()}'"
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates forecasts for the provided time series data.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the time series data with a 'unique_id' column
            and 'ds' column for timestamps.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the forecasts with columns:
            - unique_id: Unique identifier for the time series.
            - ds: Timestamps for the forecasted periods.
            - prediction: Forecasted values.
            - prediction-lo-{level}: Lower bound of the prediction interval.
            - prediction-hi-{level}: Upper bound of the prediction interval.
        """

        if not self.is_fitted:
            raise RuntimeError("The Forecaster must be fitted before predicting.")

        def _apply_forecast(X: pd.DataFrame, row: tuple[str, str, str]) -> pd.DataFrame:
            """
            Applies the forecast generation for each row of cv_results.
            """
            unique_id, _, model = row
            df = X[X[constants.UNIQUE_ID] == unique_id].copy()
            if df.empty:
                return pd.DataFrame()
            return self._generate_forecast(df, model)

        results = [
            _apply_forecast(X, row)
            for row in self.cv_results.itertuples(index=False, name=None)
        ]

        return pd.concat(results, ignore_index=True)

    def fit_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the Forecaster to the provided time series data and then generates forecasts.
        """
        return self.fit(X).predict(X)

    def _generate_forecast(self, df: pd.DataFrame, model: str) -> pd.DataFrame:
        """
        Generates a forecast for a specific unique_id using the best model.
        """
        model_list = [ModelBank().get(model)]

        sf = StatsForecast(
            models=model_list,
            freq=self.freq,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        result = sf.forecast(
            df=df,
            h=self.forecast_horizon,
            level=self.level,
        )

        result.columns = (  # type: ignore
            [
                constants.UNIQUE_ID,
                constants.TIMESTAMP_COLUMN,
                constants.PREDICTION,
            ]
            + [
                f"{constants.PREDICTION}-{constants.LOWER}-{level}"
                for level in self.level
            ]
            + [
                f"{constants.PREDICTION}-{constants.UPPER}-{level}"
                for level in self.level
            ]
        )

        return result  # type: ignore
