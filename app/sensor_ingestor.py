"""Ingestor for sensor metadata."""

import os

import pandas as pd

from app import constants as const


class SensorIngestor:
    """Class responsible for ingesting the sensor metadata from the file."""

    def __init__(self, file_path: str):
        """Initialize the SensorIngestor with the file path.

        Parameters
        ----------
        file_path : str
            Path to the sensor metadata file.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.file_path = file_path

    def transform(self) -> pd.DataFrame:
        """Transform the raw data into a DataFrame."""
        data = pd.read_excel(self.file_path, sheet_name=const.STACJE)

        # Rename columns based on the defined constants
        data = data.rename(columns=const.SENSOR_RENAME_DICT)

        # Drop unnecessary columns
        data = data[list(const.SENSOR_RENAME_DICT.values())]

        # Convert old station codes to sets
        # This assumes that the old station codes are separated by commas
        data.loc[data[const.OLD_STATION_CODE].notna(), const.OLD_STATION_CODE] = (
            data.loc[data[const.OLD_STATION_CODE].notna(), const.OLD_STATION_CODE]
            .str.split(",")
            .apply(lambda x: {s.strip() for s in x})
        )
        data.loc[data[const.OLD_STATION_CODE].isna(), const.OLD_STATION_CODE] = [set()] * len(
            data[data[const.OLD_STATION_CODE].isna()]
        )

        # Coalesce the station_codes
        # The sets of old stations codes are coalesced into a single set
        coalesced_data = data.groupby(
            [
                const.STATION_CODE,
                const.STATION_TYPE,
                const.AREA_TYPE,
                const.STATION_KIND,
                const.LATITUDE,
                const.LONGITUDE,
                const.PROVINCE,
                const.CITY,
            ],
            as_index=False,
        ).agg(
            {
                const.SENSOR_ID: lambda x: x.iloc[0],
                const.OLD_STATION_CODE: lambda x: set.union(*x),
            }
        )

        return coalesced_data
