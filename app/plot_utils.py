"""Plotting utilities"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from app import constants


def plot_voronoi(gdf: gpd.GeoDataFrame, variable: str, year: str):
    """Plot Voronoi diagram for a given variable and year.
    Function doesn't return anything, it just plots the Voronoi diagram.
    It is designed to work with interact from ipywidgets.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Voronoi geometries.
    variable : str
        The variable for which the Voronoi diagram is to be plotted.
    year : str
        The year for which the Voronoi diagram is to be plotted.
    """
    colname = f"{variable}_{year}_{constants.VORONOI_GEOMETRY}"

    if colname not in gdf.columns:
        print(f"Brak danych dla {variable} w {year}")
        return

    gdf = gdf[(gdf[colname].notnull()) & (gdf[colname] != "")]

    fig, ax = plt.subplots(figsize=(8, 8))

    voronoi_gdf = gdf[[colname]]
    voronoi_gdf = voronoi_gdf.set_geometry(colname)
    voronoi_gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue", alpha=0.5)

    gdf.set_geometry(constants.GEOMETRY).plot(ax=ax, color="red", markersize=5)

    ax.set_title(f"Zmienna: {variable}, Rok: {year}")
    ax.set_axis_off()

    fig.tight_layout()
    plt.show()


def plot_voronoi_area_boxplot(
    gdf: gpd.GeoDataFrame, variable: str, min_year: int, max_year: int
):
    """Plot a boxplot of Voronoi cell areas for a given variable over a range of years.
    The function does not return anything, it just plots the boxplot.
    It is designed to work with interact from ipywidgets.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Voronoi geometries.
    variable : str
        The variable for which the Voronoi cell areas are to be plotted.
    min_year : int
        The minimum year for the range of years to be considered.
    max_year : int
        The maximum year for the range of years to be considered.
    """
    years = np.arange(min_year, max_year + 1)
    area_data = {}

    for year in years:
        colname = f"{variable}_{year}_{constants.VORONOI_GEOMETRY}"

        if colname not in gdf.columns:
            area_data[year] = []
            continue

        valid_gdf = gdf[gdf[colname].notnull() & (gdf[colname] != "")]
        if valid_gdf.empty:
            area_data[year] = []
            continue

        valid_gdf = valid_gdf.set_geometry(colname)
        area_series = valid_gdf.geometry.area
        area_data[year] = area_series

    if not area_data:
        print(f"Brak danych do wykresu dla zmiennej {variable}.")
        return

    # Tworzenie boxplotu
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(area_data.values(), labels=area_data.keys(), patch_artist=True)  # type: ignore

    ax.set_title(f"Rozkład powierzchni komórek Voronoi dla zmiennej '{variable}'")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Powierzchnia")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years - 1, rotation=45)
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_voronoi_area_timeseries(
    gdf: gpd.GeoDataFrame, variable: str, min_year: int, max_year: int
):
    """Plot a time series of Voronoi cell areas for a given variable over a range of years.
    The function does not return anything, it just plots the time series.
    It is designed to work with interact from ipywidgets.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Voronoi geometries.
    variable : str
        The variable for which the Voronoi cell areas are to be plotted.
    min_year : int
        The minimum year for the range of years to be considered.
    max_year : int
        The maximum year for the range of years to be considered.
    """
    years = np.arange(min_year, max_year + 1)

    stats = {"year": [], "root_mean_square": [], "min": [], "max": [], "count": []}

    for year in years:
        colname = f"{variable}_{year}_{constants.VORONOI_GEOMETRY}"

        if colname not in gdf.columns:
            stats["year"].append(year)
            stats["root_mean_square"].append(np.nan)
            stats["min"].append(np.nan)
            stats["max"].append(np.nan)
            stats["count"].append(0)
            continue

        valid_gdf = gdf[gdf[colname].notnull() & (gdf[colname] != "")]
        if valid_gdf.empty:
            stats["year"].append(year)
            stats["root_mean_square"].append(np.nan)
            stats["min"].append(np.nan)
            stats["max"].append(np.nan)
            stats["count"].append(0)
            continue

        valid_gdf = valid_gdf.set_geometry(colname)
        area_series = valid_gdf.geometry.area

        stats["year"].append(year)
        stats["root_mean_square"].append(np.sqrt(np.mean(area_series**2)))
        stats["min"].append(area_series.min())
        stats["max"].append(area_series.max())
        stats["count"].append(len(area_series))

    if not stats["year"]:
        print(f"Brak danych do wykresu dla zmiennej {variable}.")
        return

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))

    years = stats["year"]
    root_mean_square_values = stats["root_mean_square"]
    min_values = stats["min"]
    max_values = stats["max"]
    counts = stats["count"]

    # Linie
    ax.plot(
        years,
        root_mean_square_values,
        label="Pierwiastek średniej kwadratów",
        color="blue",
        linewidth=2,
    )
    ax.plot(years, min_values, linestyle="--", label="Min", color="green", alpha=0.7)
    ax.plot(years, max_values, linestyle="--", label="Max", color="red", alpha=0.7)

    # Punkty ze zmienną wielkością
    sizes = [10 + c * 0.5 for c in counts]  # dynamiczne skalowanie
    ax.scatter(
        years,
        root_mean_square_values,
        s=sizes,
        color="blue",
        alpha=0.6,
        edgecolor="black",
        label="Średnia (punkty)",
    )

    # Stylizacja
    ax.set_title(
        f"Średnia, min i max powierzchnia komórek Voronoi dla zmiennej '{variable}'"
    )
    ax.set_xlabel("Rok")
    ax.set_ylabel("Powierzchnia")
    ax.set_xticks(np.arange(min_year, max_year + 1))
    ax.set_xticklabels(np.arange(min_year, max_year + 1), rotation=45)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()
