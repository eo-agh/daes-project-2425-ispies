"""Filter out time series with high percentage of missing values."""

from __future__ import annotations

import pandas as pd

from app import constants


class MissingValuesPercentageFilter:
    """Filter out the time series with missing values
    percentage above a specified threshold.
    """

    def __init__(self, threshold: float = 0.05):
        """
        Initialize the filter with a threshold.

        Parameters
        ----------
        threshold : float
            The maximum allowed percentage of missing values in a time series.
            Default is 0.05 (5%).
        """

        self.threshold = threshold

    def fit(self, X: pd.DataFrame) -> MissingValuesPercentageFilter:
        """Empty fit method for compatibility.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing time series data.

        Returns
        -------
        MissingValuesPercentageFilter
            The fitted filter instance.
        """
        return self

    def transform(self, X: pd.DataFrame, variable: str) -> pd.DataFrame:
        """Filter out time series with high percentage of missing values.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing time series data.
        variable : str
            The variable name for which the filter is applied.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with time series that have missing values
            percentage below the threshold.
            The output is in specific schema with columns:
            - 'year': Year of the time series
            - 'unique_id': Unique identifier for the time series
            - variable: Percentage of missing values for the variable
        """
        simple_X = X[[constants.TIMESTAMP_COLUMN, constants.UNIQUE_ID, variable]]
        simple_X[constants.YEAR] = simple_X[constants.TIMESTAMP_COLUMN].dt.year

        values_percentage = simple_X.groupby(
            [constants.YEAR, constants.UNIQUE_ID], as_index=False
        ).agg({variable: lambda x: x.isna().mean()})

        valid_curves = values_percentage[values_percentage[variable] < self.threshold]

        return valid_curves

    def fit_transform(self, X: pd.DataFrame, variable: str) -> pd.DataFrame:
        """Fit the filter and transform the data."""
        return self.fit(X).transform(X, variable)
