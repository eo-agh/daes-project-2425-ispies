"""Imputers for handling missing data in datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics.progress import ProgressBar

from app import constants


class Imputer(ABC):
    """Base class for imputers."""

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> Imputer:
        """Fit the imputer to the data.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the data to be imputed.

        Returns
        -------
        Imputer
            The fitted imputer.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame by filling missing values.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the data to be imputed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with missing values filled.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the imputer and transform the DataFrame.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the data to be imputed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with missing values filled.
        """
        return self.fit(X).transform(X)

    def _validate_dataframe(self, X: pd.DataFrame) -> None:
        """Validate that the DataFrame contains the required columns.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame to validate.

        Raises
        ------
        ValueError
            If the DataFrame does not contain the required columns.
        """
        required_columns = {constants.TIMESTAMP_COLUMN, constants.UNIQUE_ID}

        if not required_columns.issubset(X.columns):
            raise ValueError(
                f"DataFrame must contain '{constants.TIMESTAMP_COLUMN}'"
                f" and '{constants.UNIQUE_ID}' columns."
            )

        if len(X.columns) != 3:
            raise ValueError(
                "DataFrame must contain exactly three columns: "
                f"{constants.TIMESTAMP_COLUMN}, {constants.UNIQUE_ID},"
                " and the variable to be imputed."
            )

        if X.columns[-1] in required_columns:
            raise ValueError(
                f"The last column must be the variable to be imputed, "
                "not one of the required columns:"
                f" {constants.TIMESTAMP_COLUMN}, {constants.UNIQUE_ID}."
            )


class NearestSensorImputer(Imputer):
    """Imputer that fills missing values in a variable using the nearest sensor's value."""

    def __init__(self, distance_matrix: List[List[int]], sensor_ids: List[int]):
        """Initialize the imputer with a distance matrix.

        Parameters
        ----------
        distance_matrix : List[List[int]]
            A matrix representing distances between sensors.
        sensor_ids : List[int]
            A list of sensor IDs corresponding to the distance matrix.
        """
        if len(distance_matrix) != len(sensor_ids):
            raise ValueError("Distance matrix size must match the number of sensor IDs.")

        self.distance_matrix = distance_matrix
        self.sensor_ids = sensor_ids

        self.is_fitted = False
        self.nearest_sensors = None

    def fit(self, X: pd.DataFrame) -> NearestSensorImputer:
        """Empty fit method for compatibility.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the data to be imputed.
            The data should contain TIMESTAMP COLUMN, SENSOR ID COLUMN,
            and the single variable to be imputed.

        Returns
        -------
        NearestSensorImputer
            The fitted imputer.
        """
        self.nearest_sensors = self._create_sensor_distance_dict()
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame by filling missing values.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the data to be imputed.
            The data should contain TIMESTAMP COLUMN, SENSOR ID COLUMN,
            and the single variable to be imputed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with missing values filled.
        """
        if not self.is_fitted:
            raise RuntimeError("Imputer must be fitted before transformation.")

        self._validate_dataframe(X)

        def _impute_nearest_partition(
            df: pd.DataFrame,
            X_series: pd.Series,
            nearest_sensors: dict,
            variable_column: str,
        ) -> pd.DataFrame:
            """Impute missing values for a partition of the Dask DataFrame."""

            def _impute_row(row):
                """Impute a single row by finding the nearest sensor value."""
                index = 0
                measure_date = row[constants.TIMESTAMP_COLUMN]
                sensor_id = row[constants.UNIQUE_ID]
                value = row[variable_column]
                while pd.isna(value) and index < len(nearest_sensors[sensor_id]):
                    try:
                        value = X_series.loc[(measure_date, nearest_sensors[sensor_id][index])]
                    except KeyError:
                        pass
                    index += 1
                return value

            df[variable_column] = df.apply(_impute_row, axis=1)
            return df

        variable_column = X.columns[-1]
        X_series = X.set_index([constants.TIMESTAMP_COLUMN, constants.UNIQUE_ID])[variable_column]
        X_dask = dd.from_pandas(X, npartitions=None)

        with ProgressBar():
            X_imputed = X_dask.map_partitions(
                _impute_nearest_partition,
                X_series=X_series,
                nearest_sensors=self.nearest_sensors,
                variable_column=variable_column,
                meta=X,
            ).compute()

        return X_imputed

    def _create_sensor_distance_dict(self) -> dict[int, List[int]]:
        """Create a dictionary mapping each sensor ID
        to a list of other sensor IDs sorted by distance.

        Returns
        -------
        dict[int, List[int]]
            A dictionary where keys are sensor IDs
            and values are lists of other sensor IDs sorted by distance.
        """
        sensor_distance_dict = {}
        for i, sensor_id in enumerate(self.sensor_ids):
            sorted_indices = np.argsort(self.distance_matrix[i])
            sorted_sensor_ids = [int(self.sensor_ids[j]) for j in sorted_indices]
            sensor_distance_dict[int(sensor_id)] = sorted_sensor_ids
        return sensor_distance_dict


class SupportedLastImputer(Imputer):
    """Imputer that fills missing values with the last known value"""

    def __init__(self, support_imputer: Imputer):
        """Initialize the imputer."""
        self.is_fitted = False
        self.support_imputer = support_imputer

    def fit(self, X: pd.DataFrame) -> SupportedLastImputer:
        """Empty fit method for compatibility.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the data to be imputed.
            The data should contain TIMESTAMP COLUMN, SENSOR ID COLUMN,
            and the single variable to be imputed.

        Returns
        -------
        SupportedLastImputer
            The fitted imputer.
        """
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame by filling missing values with the last known value.
        The first and last indexes of the DataFrame are filled with the support imputer.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the data to be imputed.
            The data should contain TIMESTAMP COLUMN, SENSOR ID COLUMN,
            and the single variable to be imputed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with missing values filled.
        """
        if not self.is_fitted:
            raise RuntimeError("Imputer must be fitted before transformation.")

        self._validate_dataframe(X)

        X = X.sort_values([constants.UNIQUE_ID, constants.TIMESTAMP_COLUMN]).reset_index(drop=True)

        def _impute_time_series(X: pd.DataFrame, unique_id: int) -> pd.DataFrame:
            """Impute missing values for a single time series."""
            variable_column = X.columns[-1]

            time_series = X[X[constants.UNIQUE_ID] == unique_id].reset_index(drop=True)
            first_valid_index = time_series[variable_column].first_valid_index()
            last_valid_index = time_series[variable_column].last_valid_index()

            data_for_support_imputer_first = time_series.iloc[:first_valid_index, :]
            data_for_support_imputer_last = time_series.iloc[last_valid_index:, :]
            data_for_imputation = time_series.iloc[first_valid_index:last_valid_index, :]

            result_first = self._use_support_imputer(X, data_for_support_imputer_first, unique_id)
            result_last = self._use_support_imputer(X, data_for_support_imputer_last, unique_id)
            result_middle = data_for_imputation.ffill()

            return pd.concat([result_first, result_middle, result_last], ignore_index=True)

        results = [
            _impute_time_series(X, unique_id) for unique_id in X[constants.UNIQUE_ID].unique()
        ]

        return pd.concat(results, ignore_index=True)

    def _use_support_imputer(
        self, X: pd.DataFrame, time_series: pd.DataFrame, unique_id: int
    ) -> pd.DataFrame:
        """Use the support imputer to fill missing values in the DataFrame."""
        if time_series.empty:
            return pd.DataFrame()

        data = X[X[constants.TIMESTAMP_COLUMN].isin(time_series[constants.TIMESTAMP_COLUMN])]

        result = self.support_imputer.fit_transform(data)
        result = result.query(f"{constants.UNIQUE_ID} == {unique_id}")

        return result
