"""Transformer that produces Voronoi polygons for given data."""

from __future__ import annotations

from typing import List

import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

from app import constants


class VoronoiTransformer:
    """Transformer that produces Voronoi polygons for given data."""

    def __init__(
        self,
        mask_polygon: gpd.GeoDataFrame,
        buffer_size: float = constants.BUFFER_SIZE,
        buffer_points_amount: int = constants.BUFFER_POINTS_AMOUNT,
    ):
        """Initialize the VoronoiTransformer.

        Parameters
        ----------
        mask_polygon : gpd.GeoDataFrame
            GeoDataFrame containing the polygon that serves as a mask for the Voronoi polygons.
            This polygon should be in the same CRS as the data to be transformed.
        buffer_size : float
            Size of the buffer around the mask polygon.
        buffer_points_amount : int
            Number of points to create based on the buffer polygon.
        """
        self.mask_polygon = mask_polygon
        self.buffer_size = buffer_size
        self.buffer_points_amount = buffer_points_amount

        self.is_fitted = False
        self.buffer_points = None

    def fit(self, X: gpd.GeoDataFrame) -> VoronoiTransformer:
        """Fit the transformer to the data.

        Parameters
        ----------
        X : gpd.GeoDataFrame
            GeoDataFrame containing the points for which Voronoi polygons are created.

        Returns
        -------
        VoronoiTransformer
            The fitted transformer with buffer points calculated.
        """
        buffer = self.mask_polygon.buffer(self.buffer_size)
        buffer_boundary = buffer.boundary
        buffer_boundary_length = buffer_boundary.length.values[0]
        distances = np.linspace(0, buffer_boundary_length, self.buffer_points_amount)
        points = [buffer_boundary.interpolate(d).values[0] for d in distances]

        self.buffer_points = gpd.GeoDataFrame(geometry=points, crs=self.mask_polygon.crs)
        self.is_fitted = True
        return self

    def transform(self, X: gpd.GeoDataFrame) -> gpd.GeoSeries:
        """Transform the data to Voronoi polygons.

        Parameters
        ----------
        X : gpd.GeoDataFrame
            GeoDataFrame containing the points for which Voronoi polygons are created.

        Returns
        -------
        gpd.GeoSeries
            A GeoSeries containing the Voronoi polygons for the input points.
        """
        if not self.is_fitted:
            raise RuntimeError("The transformer must be fitted before transforming data.")

        X_to_process = X.copy()
        if X_to_process.crs != self.mask_polygon.crs:
            X_to_process = X_to_process.to_crs(self.mask_polygon.crs)  # type: ignore

        points_sensors = np.array(
            [[geom.x, geom.y] for geom in X_to_process.geometry]  # type: ignore
        )
        points_buffer = np.array(
            [[geom.x, geom.y] for geom in self.buffer_points.geometry]  # type: ignore
        )
        points = np.concatenate([points_sensors, points_buffer], axis=0)

        try:
            vor = Voronoi(points)
        except ValueError as e:
            raise ValueError("Voronoi diagram could not be created.") from e

        polygons = self._voronoi_to_polygons(vor=vor, initial_points_length=len(points_sensors))

        mask_polygon_geometry = self.mask_polygon.geometry.values[0]
        polygons = [polygon.intersection(mask_polygon_geometry) for polygon in polygons]

        return gpd.GeoSeries(
            polygons,
            crs=self.mask_polygon.crs,
        )

    def fit_transform(self, X: gpd.GeoDataFrame) -> gpd.GeoSeries:
        """Fit the transformer and transform the data to Voronoi polygons."""
        return self.fit(X).transform(X)

    @staticmethod
    def _voronoi_to_polygons(vor: Voronoi, initial_points_length: int) -> List[Polygon]:
        """Convert Voronoi regions to polygons.

        Parameters
        ----------
        vor : scipy.spatial.Voronoi
            The Voronoi object containing the regions.
        initial_points_length : int
            The number of initial points used to create the Voronoi diagram.
            In the other words, the number of points that were not buffer points.

        Returns
        -------
        List[Polygon | None]
            A list of polygons representing the Voronoi regions.
            If a region is invalid (e.g., contains -1), it returns None for that region.
        """
        regions = []

        for i in range(len(vor.points)):
            if i >= initial_points_length:
                break

            region_index = vor.point_region[i]
            region = vor.regions[region_index]

            if not region or -1 in region:
                regions.append(Polygon())
                continue

            polygon = Polygon([vor.vertices[j] for j in region])
            regions.append(polygon)

        return regions
