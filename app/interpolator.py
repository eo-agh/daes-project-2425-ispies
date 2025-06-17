"""Interpolator class for handling interpolation methods."""

from __future__ import annotations

import copy
import warnings
from typing import Callable

import geopandas as gpd
from scipy.interpolate import Rbf
from sklearn.neighbors import KNeighborsClassifier

from app import constants
from app.data_types import METRIC_BANK

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Interpolator:
    """
    Interpolator class for applying various interpolation methods to GeoDataFrames
    and calculating metrics based on the interpolation results.
    """

    def __init__(
        self,
        method: Callable[[gpd.GeoDataFrame, gpd.GeoDataFrame], gpd.GeoDataFrame],
        points: gpd.GeoDataFrame,
    ):
        """Initialize the Interpolator with a specific interpolation method.

        Parameters
        ----------
        method : Callable[
                [gpd.GeoDataFrame, gpd.GeoDataFrame],
                gpd.GeoDataFrame
            ]
            A function that takes two GeoDataFrames and
            returns a GeoDataFrame after applying the interpolation.
        points : gpd.GeoDataFrame
            GeoDataFrame containing the points to be used for interpolation.
            These points should be in the same CRS as the data to be transformed.
        """
        self.method = method
        self.points = points
        self._metrics: dict[str, float] = {}

    @property
    def metrics(self) -> dict[str, float]:
        """Get the metrics calculated during the transformation.

        Returns
        -------
        dict[str, float]
            A dictionary containing the metrics calculated during the transformation.
        """
        return copy.deepcopy(self._metrics)

    def fit(self, X: gpd.GeoDataFrame) -> Interpolator:
        """Fit the interpolator to the data.

        Parameters
        ----------
        X : gpd.GeoDataFrame
            GeoDataFrame for which the interpolation is to be fitted.

        Returns
        -------
        Interpolator
            The fitted interpolator.
        """
        return self

    def transform(self, X: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Transform the data using the fitted interpolation method.

        Parameters
        ----------
        X : gpd.GeoDataFrame
            GeoDataFrame containing the data to be transformed,
            the same as used in fit.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the transformed data with interpolated values.
        """
        pairs = [self._prediction_for_missing_point(index, X) for index in self.points.index]

        metrics = {
            key: value(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
            )
            for key, value in METRIC_BANK.items()
        }

        result = self.method(self.points, X)

        self._metrics = metrics
        return result

    def fit_transform(self, X: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fit the interpolator and then transform the data.

        Parameters
        ----------
        X : gpd.GeoDataFrame
            GeoDataFrame containing the data to be transformed.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the transformed data with interpolated values.
        """
        return self.fit(X).transform(X)

    def _prediction_for_missing_point(self, index: int, X: gpd.GeoDataFrame) -> tuple[float, float]:
        """Use N-1 points to predict the value for a missing point."""
        point = self.points.loc[[index]]
        points = self.points.drop(index)
        result = self.method(points, X)
        nearest_index = result.distance(point.geometry.iloc[0]).idxmin()  # type: ignore
        nearest_value = result.loc[nearest_index, constants.Y]
        return point[constants.Y], nearest_value  # type: ignore


def knn_one_euclidean_method(
    points: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """KNN interpolation method.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        GeoDataFrame containing the points to be used for interpolation.
    grid : gpd.GeoDataFrame
        GeoDataFrame containing the grid points where the interpolation is applied.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the interpolated values at the grid points.
    """
    grid = grid.copy()
    classifier = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    classifier.fit(
        points[[constants.LATITUDE, constants.LONGITUDE]].to_numpy(),
        points[constants.UNIQUE_ID],
    )
    prediction = classifier.predict(grid[[constants.LATITUDE, constants.LONGITUDE]].to_numpy())
    grid[constants.UNIQUE_ID] = prediction
    merged = grid.merge(
        points[[constants.UNIQUE_ID, constants.Y]],
        left_on=constants.UNIQUE_ID,
        right_on=constants.UNIQUE_ID,
        how="left",
    )
    return merged


def thin_plate_spline_method(
    points: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Thin plate spline interpolation method.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        GeoDataFrame containing the points to be used for interpolation.
    grid : gpd.GeoDataFrame
        GeoDataFrame containing the grid points where the interpolation is applied.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the interpolated values at the grid points.
    """
    grid = grid.copy()
    rbf = Rbf(
        points[constants.LATITUDE],
        points[constants.LONGITUDE],
        points[constants.Y],
        function="thin_plate",
    )
    grid[constants.Y] = rbf(grid[constants.LATITUDE], grid[constants.LONGITUDE])
    return grid
