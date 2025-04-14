import os

import pandas as pd

STACJE = "STACJE"
SENSOR_ID = "sensor_id"
STATION_CODE = "station_code"
OLD_STATION_CODE = "old_station_code"
STATION_TYPE = "station_type"
AREA_TYPE = "area_type"
STATION_KIND = "station_kind"
LATITUDE = "latitude"
LONGITUDE = "longitude"
PROVINCE = "province"
CITY = "city"

SENSOR_RENAME_DICT = {
    "Nr": SENSOR_ID,
    "Kod stacji": STATION_CODE,
    "Stary Kod stacji \n(o ile inny od aktualnego)": OLD_STATION_CODE,
    "Typ stacji": STATION_TYPE,
    "Typ obszaru": AREA_TYPE,
    "Rodzaj stacji": STATION_KIND,
    "WGS84 φ N": LATITUDE,
    "WGS84 λ E": LONGITUDE,
    "Województwo": PROVINCE,
    "Miejscowość": CITY,
}


class SensorIngestor:
    """Class responsible for ingesting the sensor metadata from the file."""

    def __init__(self, file_path: str):
        """Initialize the SensorIngestor with the file path.

        Parameters
        ----------
        file_path : str
            Path to the sensor metadata file.
        """
        self.file_path = file_path
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

    def transform(self) -> pd.DataFrame:
        """Transform the raw data into a DataFrame."""
        data = pd.read_excel(self.file_path, sheet_name=STACJE)
        data = data.rename(columns=SENSOR_RENAME_DICT)
        data = data[list(SENSOR_RENAME_DICT.values())]
        data.loc[data[OLD_STATION_CODE].notna(), OLD_STATION_CODE] = (
            data.loc[data[OLD_STATION_CODE].notna(), OLD_STATION_CODE]
            .str.split(",")
            .apply(set)
        )
        return data
