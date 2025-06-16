"""CrossValidator for time series models."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from statsforecast import StatsForecast

from app import constants
from app.data_types import METRIC_BANK, Metric, ModelBank


class CrossValidator:
    """
    A class for performing cross-validation on time series models.
    It determines the best model based on a specified metric.
    """

    def __init__(
        self,
        models: list[str],
        freq: str,
        forecast_horizon: int = 7,
        season_length: int = 365,
        cv_folds: int = 5,
        metric: Metric = constants.MAE,  # type: ignore
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        """
        Initializes the CrossValidator with the given parameters.

        Parameters
        ----------
        models : list[str]
            List of model names to be used for cross-validation.
        freq : str
            Frequency of the time series data (e.g., 'D' for daily, 'H' for hourly).
        forecast_horizon : int, optional
            The number of periods to forecast ahead (default is 7).
        season_length : int, optional
            The length of the seasonal cycle (default is 365).
        cv_folds : int, optional
            The number of cross-validation folds (default is 5).
        metric : Metric, optional
            The metric to evaluate model performance (default is constants.MAE).
        n_jobs : int, optional
            Level of parallelism for cross-validation (default is 1).
        verbose : bool, optional
            Whether to print verbose output (default is True).
        """
        self.models = models
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.cv_folds = cv_folds
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.sf = None
        self.metric_callable = None
        self.is_fitted = False

        self._error_df = None

    @property
    def error_df(self) -> pd.DataFrame:
        """
        Returns the error DataFrame containing the cross-validation results.
        This property is only available after the CrossValidator has been fitted and transformed.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the cross-validation results, including unique identifiers,
            start dates, and the best model determined by the cross-validation process.
        """
        if self._error_df is None:
            raise RuntimeError(
                "The CrossValidator has not been transformed yet. Call transform() first."
            )
        return self._error_df.copy()

    def fit(self, X: pd.DataFrame) -> CrossValidator:
        """
        Fits the CrossValidator to the provided time series data.

        Parameters
        ----------
        X : pd.DataFrame
            The time series data to fit the CrossValidator on.
            It's unused in the fitting process, for compatibility with the interface.

        Returns
        -------
        CrossValidator
            The fitted CrossValidator instance.
        """
        model_instances = [ModelBank().get(model_name) for model_name in self.models]
        self.metric_callable = METRIC_BANK.get(self.metric)
        self.sf = StatsForecast(
            models=model_instances,
            freq=self.freq,
            n_jobs=self.n_jobs,
            verbose=True,
        )

        self.is_fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the provided time series data by performing cross-validation
        and determining the best model based on the specified metric.

        Parameters
        ----------
        X : pd.DataFrame
            The time series data to transform. It should contain the necessary columns
            for cross-validation, including unique identifiers and timestamps.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the unique identifiers, start dates, and the best model
            determined by the cross-validation process.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "The CrossValidator must be fitted before transforming data."
            )

        cv_df: pd.DataFrame = self.sf.cross_validation(  # type: ignore
            df=X,
            h=self.forecast_horizon,
            step_size=self.season_length // self.cv_folds,
            n_windows=self.cv_folds,
        )

        return self._determine_best_model(cv_df)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the CrossValidator to the provided time series data and then transforms it
        by performing cross-validation and determining the best model.

        Parameters
        ----------
        X : pd.DataFrame
            The time series data to fit and transform. It should contain the necessary columns
            for cross-validation, including unique identifiers and timestamps.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the unique identifiers, start dates, and the best model
            determined by the cross-validation process.
        """
        return self.fit(X).transform(X)

    def _determine_best_model(self, cv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Determines the best model based on the specified metric.
        """

        def _calculate_metrics(
            group: pd.DataFrame,
            metric: str,
            metric_callabe: Callable[[pd.Series, pd.Series], float],
            models: list[str],
        ) -> pd.Series:
            """Calculates the specified metric for each model in the group."""
            result = pd.Series(
                {
                    f"{metric}_{model}": metric_callabe(
                        group[constants.Y], group[model]
                    )
                    for model in models
                }
            )
            return result

        cv_df_grouped = cv_df.groupby([constants.UNIQUE_ID, constants.CUTOFF])

        cv_df_results = cv_df_grouped.apply(
            lambda group: _calculate_metrics(
                group=group,
                metric=self.metric,
                metric_callabe=self.metric_callable,  # type: ignore
                models=self.models,
            )
        ).reset_index()

        cv_df_results[constants.BEST_MODEL] = (
            cv_df_results.filter(like=self.metric)
            .idxmin(axis=1)
            .str.replace(f"{self.metric}_", "")
        )

        self._error_df = cv_df_results.copy()

        cv_df_results[constants.CUTOFF] = pd.to_datetime(
            cv_df_results[constants.CUTOFF], format=constants.CUTOFF_FORMAT
        ) + pd.Timedelta(
            self.forecast_horizon, unit=self.freq  # type: ignore
        )
        cv_df_results = cv_df_results.rename(
            columns={constants.CUTOFF: constants.START_DATE}
        )

        return cv_df_results[
            [constants.UNIQUE_ID, constants.START_DATE, constants.BEST_MODEL]
        ]
