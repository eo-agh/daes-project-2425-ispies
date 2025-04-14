"""Ingesor for measurement data."""

import logging
import os
from ast import Index
from datetime import datetime
from functools import reduce
from re import sub
from typing import DefaultDict, Dict, List, Literal, Optional, Tuple

import pandas as pd

from app.sensor_ingestor import OLD_STATION_CODE, SENSOR_ID, STATION_CODE

MeasureTime = Literal["1g", "24g"]

COLUMN_0 = "0"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

KOD_STACJI = "Kod stacji"
CZAS_USREDNIANIA = "Czas uśredniania"
WSKAZNIK = "Wskaźnik"

DEPOZYCJA = "depozycja"

TIME_1H = "1g"
TIME_24H = "24g"

TIMESTAMP_COLUMN = "ds"
UNIQUE_ID = "unique_id"


class MeasurementIngestor:
    """Class responsible for ingesting the measurement data from the directory.

    The output of the transform process is a tuple of two DataFrames:
    - The first DataFrame contains the 1-hour measurements.
    - The second DataFrame contains the 24-hour measurements.

    The example of the output DataFrame is:
    unique_id  | ds                  | variable_1 | variable_2 | ... | variable_n
    -----------|---------------------|------------|------------|-----|------------
    10         | 2023-01-01 00:00:00 | 1.0        | 2.0        | ... | n.0
    10         | 2023-01-01 01:00:00 | 1.1        | 2.1        | ... | n.1
    10         | 2023-01-01 02:00:00 | 1.2        | 2.2        | ... | n.2
    20         | 2023-01-01 00:00:00 | 1.0        | 2.0        | ... | n.0
    20         | 2023-01-01 01:00:00 | 1.1        | 2.1        | ... | n.1
    """

    def __init__(
        self,
        input_dir: str,
        sensor_metadata: pd.DataFrame,
        exclude_depoyzcja: bool = True,
        target_variables: Optional[List[str]] = None,
    ):
        """Initialize the MeasurementIngestor with the input directory.

        Parameters
        ----------
        input_dir : str
            The path to the directory containing the measurement data.
            The directory should contain subdirectories with the measurement files.
        sensor_metadata : pd.DataFrame
            The DataFrame containing the sensor metadata.
            It is output of the SensorIngestor class.
        exclude_depoyzcja : bool, optional
            Whether to exclude the files containing "depoyzcja" in their names.
            Default is True.
        target_variables : List[str], optional
            The list of target variables to process.
            If None, all variables will be processed.
            Default is None.
        """
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Directory {input_dir} does not exist.")
        self.input_dir = input_dir
        self.sensor_metadata = sensor_metadata
        self.exclude_depoyzcja = exclude_depoyzcja
        self.target_variables = target_variables

    def transform(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform the measurement data from the input directory into a DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing two DataFrames:
            - The first DataFrame contains the 1-hour measurements.
            - The second DataFrame contains the 24-hour measurements.
        """
        time_1h = DefaultDict(list)
        time_24h = DefaultDict(list)

        sub_directories = os.listdir(self.input_dir)
        files = [
            os.path.join(self.input_dir, sub_dir, file)
            for sub_dir in sub_directories
            for file in os.listdir(os.path.join(self.input_dir, sub_dir))
        ]

        for file in files:
            # Extract subdirectory name from the file path
            sub_dir = os.path.basename(os.path.dirname(file))

            # Check if the file is excluded from the processing
            if self.exclude_depoyzcja and DEPOZYCJA in file.lower():
                logging.info(
                    "File %s is excluded from the processing. Check parameter exclude_depozycja=%s.",
                    file,
                    self.exclude_depoyzcja,
                )
                continue

            # Check if the file is an Excel file
            if not file.endswith(".xlsx"):
                logging.warning("File %s is not an Excel file. Skipping.", file)
                continue

            # Read the Excel file
            single_measure_df = pd.read_excel(file, header=None)

            # Convert columns to string
            single_measure_df.columns = [str(col) for col in single_measure_df.columns]

            # Read measurement metadata
            try:
                measure_time, variable = self._scrap_measurement_metadata(
                    single_measure_df
                )
            except IndexError:
                logging.warning(
                    "Invalid measurement metadata in file %s. Skipping.", file
                )
                continue

            if self.target_variables and variable not in self.target_variables:
                continue

            if not measure_time:
                logging.warning("Invalid measurement time in file %s. Skipping.", file)
                continue

            # Scrap the sensor names
            sensor_names = self._scrap_sensor_names(single_measure_df)

            # Find the first date index
            first_date_index = self._scrap_first_date_index(single_measure_df)
            if not first_date_index:
                logging.warning("No date index found in file %s. Skipping.", file)
                continue

            # Filter the DataFrame to keep only the relevant data
            single_measure_df = single_measure_df.iloc[int(first_date_index) :]

            # Create dict to map the column names to sensor unique IDs
            sensor_codes = self._bind_sensor_name_with_sensor_unique_id(sensor_names)

            # Rename the columns to match the sensor unique IDs
            # and remove the columns with undefined sensors
            single_measure_df = self._transform_df_to_select_proper_columns(
                single_measure_df, sensor_codes
            )

            # Melt data to the nixtla based wide format
            melted_single_measure_df = single_measure_df.melt(
                id_vars=[COLUMN_0], var_name=UNIQUE_ID, value_name=variable
            )

            # Rename date column
            melted_single_measure_df = melted_single_measure_df.rename(
                columns={COLUMN_0: TIMESTAMP_COLUMN}
            )

            # Adjust types of the columns
            melted_single_measure_df[TIMESTAMP_COLUMN] = pd.to_datetime(
                melted_single_measure_df[TIMESTAMP_COLUMN], format=DATE_FORMAT
            )
            melted_single_measure_df[UNIQUE_ID] = melted_single_measure_df[
                UNIQUE_ID
            ].astype(int)
            melted_single_measure_df[variable] = melted_single_measure_df[
                variable
            ].astype(float)

            # Assign the measure to the correct list
            # It cannot happen here that the measure time is not 1h or 24h
            if measure_time == TIME_1H:
                time_1h[sub_dir].append(melted_single_measure_df)
            elif measure_time == TIME_24H:
                time_24h[sub_dir].append(melted_single_measure_df)

        df_time_1h = self._concat_df(time_1h)
        df_time_24h = self._concat_df(time_24h)

        df_time_1h = self._resample_df(df_time_1h, TIME_1H)
        df_time_24h = self._resample_df(df_time_24h, TIME_24H)

        return df_time_1h, df_time_24h

    def _scrap_measurement_metadata(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[MeasureTime], str]:
        """Scrap the measurement metadata from the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the measurement data.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the measurement time and the variable name.
        """
        measure_time = (
            data.query(f"`{COLUMN_0}` == '{CZAS_USREDNIANIA}'").iloc[:, 1].values[0]
        )
        variable = data.query(f"`{COLUMN_0}` == '{WSKAZNIK}'").iloc[:, 1].values[0]

        # Check measure time
        if measure_time != TIME_1H and measure_time != TIME_24H:
            measure_time = None
        return measure_time, variable

    def _scrap_sensor_names(self, data: pd.DataFrame) -> Dict[str, str]:
        """Scrap the sensor names from the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the measurement data.

        Returns
        -------
        List[str]
            A list of sensor names.
        """
        sensor_names = data.query(f"`{COLUMN_0}` == '{KOD_STACJI}'").iloc[0, 1:]
        return sensor_names.to_dict()

    def _scrap_first_date_index(self, data: pd.DataFrame) -> Optional[int]:
        """Scrap the first date index from the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the measurement data.

        Returns
        -------
        str
            The first date index.
        """
        for index, value in enumerate(data[COLUMN_0]):
            if isinstance(value, datetime):
                return index

    def _bind_sensor_name_with_sensor_unique_id(
        self, sensor_names: Dict[str, str]
    ) -> Dict[str, List[int]]:
        """Match the sensor names with their unique IDs.

        Parameters
        ----------
        sensor_names : Dict[str, str]
            The dictionary containing the sensor names.
            The key is the column name and the value is the sensor name.

        Returns
        ---------
        Dict[str, List[int]]
            The dictionary containing the sensor names and their unique IDs.
            The key is the column name and the value is the sensor unique ID.
            The sensor unique ID is a list of integers.
        """
        station_codes = {}
        for key, value in sensor_names.items():
            # Check if the value is present in the old station code column
            # If it is, use the old station code to find the sensor ID
            old_station_code_mask = self.sensor_metadata[OLD_STATION_CODE].apply(
                lambda x: value in x
            )
            old_sensor_names_df = self.sensor_metadata[old_station_code_mask]
            if not old_sensor_names_df.empty:
                station_codes[key] = old_sensor_names_df[SENSOR_ID].tolist()
                station_codes[key] = [int(x) for x in station_codes[key]]
                continue

            # If the value is not present in the old station code column,
            # use the station code to find the sensor ID
            # Check if the value is present in the station code column
            station_code_mask = self.sensor_metadata[STATION_CODE] == value
            sensor_names_df = self.sensor_metadata[station_code_mask]
            if sensor_names_df.empty:
                logging.warning("Sensor name %s not found. Skipping.", value)
                continue
            if len(sensor_names_df) > 1:
                logging.warning("Multiple sensor names found for %s. Skipping.", value)
                continue
            station_codes[key] = sensor_names_df[SENSOR_ID].tolist()
            station_codes[key] = [int(x) for x in station_codes[key]]
        return station_codes

    def _transform_df_to_select_proper_columns(
        self, data: pd.DataFrame, sensor_codes: Dict[str, List[int]]
    ) -> pd.DataFrame:
        """Transform the DataFrame to select the proper columns.

        Parameters
        ---------
        data : pd.DataFrame
            The DataFrame containing the measurement data.
        sensor_codes : Dict[str, List[int]]
            The dictionary containing the sensor names and their unique IDs.
            The key is the column name and the value is the sensor unique ID.
            The sensor unique ID is a list of integers.

        Returns
        ---------
        pd.DataFrame
            The transformed DataFrame with the selected columns.
            The columns are renamed to match the sensor unique IDs.
            The columns are duplicated, if the sensor old code is reference to the
            multiple sensor unique IDs.
            The columns with undefined sensors are removed.
        """
        df = data.copy()

        # Select the columns to keep - identified sensor codes + the first column
        # The first column is the timestamp column
        cols_to_keep = [COLUMN_0] + list(sensor_codes.keys())

        # Select the columns to duplicate - identified sensor codes with multiple IDs
        columns_to_duplicate = [
            key for key, value in sensor_codes.items() if len(value) > 1
        ]

        # The columns to rename are the columns that are not duplicated
        col_to_rename = list(set(cols_to_keep) - set(columns_to_duplicate) - {COLUMN_0})

        # Create dataframe for sub operations
        # It's imporant to separate the rename and duplicate operations
        # To avoid conflicts with the column names
        df = df[cols_to_keep]

        # Rename the columns to match the sensor unique IDs
        df_to_rename = df[col_to_rename].copy()
        df_renamed = df_to_rename.rename(
            columns={col: sensor_codes[col][0] for col in col_to_rename}
        )

        # Duplicate the columns for the sensor codes with multiple IDs
        df_with_duplicated = df.loc[:, COLUMN_0].to_frame()
        for col in columns_to_duplicate:
            new_columns = sensor_codes[col]
            for new_col in new_columns:
                df_with_duplicated[new_col] = df[col].copy()

        # Return the concatenated DataFrame
        return pd.concat([df_with_duplicated, df_renamed], axis=1, ignore_index=False)

    def _concat_df(self, data: Dict[str, list]) -> pd.DataFrame:
        """Concatenate the DataFrames in the list.

        Parameters
        ----------
        data : Dict[str, list]
            The dictionary containing the DataFrames to concatenate.
            The key is the subdirectory name and the value is the list of DataFrames.

        Returns
        -------
        pd.DataFrame
            The concatenated DataFrame.
        """

        def merge_func(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
            """Heleper function to merge two DataFrames."""
            return pd.merge(x, y, how="outer", on=[TIMESTAMP_COLUMN, UNIQUE_ID])

        df = pd.concat(
            [reduce(merge_func, data[key]) for key in data.keys()],
            ignore_index=True,
        )

        return df

    def _resample_df(
        self, data: pd.DataFrame, measure_time: MeasureTime
    ) -> pd.DataFrame:
        """Resample the DataFrame to the specified measurement time.

        Parameters
        ------------
        data : pd.DataFrame
            The DataFrame containing the measurement data.
        measure_time : MeasureTime
            The measurement time to resample to.
            Can be either "1g" or "24g".

        Returns
        -------
        pd.DataFrame
            The resampled DataFrame.
        """
        result = []
        freq = "h" if measure_time == TIME_1H else "24h"
        for unique_id in data[UNIQUE_ID].unique():
            df_id = data.query(f"`{UNIQUE_ID}` == {unique_id}")
            df_id = df_id.sort_values(by=[TIMESTAMP_COLUMN], ascending=True)
            df_id = df_id.set_index(TIMESTAMP_COLUMN, drop=True)
            df_id = df_id.asfreq(freq)
            df_id[UNIQUE_ID] = unique_id
            df_id = df_id.reset_index()
            result.append(df_id)
        return pd.concat(result, ignore_index=True)
