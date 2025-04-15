"""Ingesor for measurement data."""

import logging
import os
from datetime import datetime
from functools import reduce
from typing import DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd

from app import constants as const

MeasureTime = Literal["1g", "24g"]


class MeasurementIngestor:
    """Class responsible for ingesting the measurement data from the directory.

    The output of the transform process is a tuple of two DataFrames:
    - The first DataFrame contains the 1-hour measurements.
    - The second DataFrame contains the 24-hour measurements.

    The example of the output DataFrame is:
    unique_id  | ds                  | variable_1 | variable_2 | ... | variable_n
    -----------|---------------------|------------|------------|-----|------------
    10         | 2023-01-01 00:00:00 | 1.0        | 2.0        | ... | 3.123
    10         | 2023-01-01 01:00:00 | 1.1        | 2.1        | ... | 56.1232
    10         | 2023-01-01 02:00:00 | 1.2        | 2.2        | ... | 123.412
    20         | 2023-01-01 00:00:00 | 1.0        | 2.0        | ... | 124.123
    20         | 2023-01-01 01:00:00 | 1.1        | 2.1        | ... | 3.123
    20         | 2023-01-01 02:00:00 | 1.2        | 2.2        | ... | 98.1
    """

    def __init__(
        self,
        input_dir: str,
        sensor_metadata: pd.DataFrame,
        exclude_depoyzcja: bool = True,
        target_variables: Optional[Iterable[str]] = const.TARGET_VARIABLES,
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
            Default is TARGET_VARIABLES defined in constants.py.
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
            # Extract subdirectory and file name from the file path
            sub_dir = os.path.basename(os.path.dirname(file))
            file_name = os.path.basename(file)

            # Check if the file is excluded from the processing
            if self.exclude_depoyzcja and const.DEPOZYCJA in file_name.lower():
                logging.info(
                    "File %s is excluded from the processing."
                    " Check parameter exclude_depozycja=%s.",
                    file_name,
                    self.exclude_depoyzcja,
                )
                continue

            # Extract the measurement metadata from the file name
            measure_time, variable = self._extract_measurement_metadata(file_name)

            # Skip the file if variable is not in the target variables
            if self.target_variables and variable not in self.target_variables:
                continue

            # Skip the file if measure time is not valid
            if not measure_time:
                logging.warning("Invalid measurement time in file %s. Skipping.", file_name)
                continue

            # Check if the file is an Excel file
            if not file_name.endswith(".xlsx"):
                logging.warning("File %s is not an Excel file. Skipping.", file_name)
                continue

            # Read the Excel file
            single_measure_df = pd.read_excel(file, header=None)

            # Convert columns to string
            single_measure_df.columns = [str(col) for col in single_measure_df.columns]

            # Find the first date index
            first_date_index = self._scrap_first_date_index(single_measure_df)
            if not first_date_index:
                logging.warning("No date index found in file %s. Skipping.", file_name)
                continue

            # Scrap the sensor names
            sensor_names = self._scrap_sensor_names(
                single_measure_df,
                first_date_index=first_date_index,
                file_name=file_name,
            )

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
                id_vars=[const.COLUMN_0], var_name=const.UNIQUE_ID, value_name=variable
            )

            # Rename date column
            melted_single_measure_df = melted_single_measure_df.rename(
                columns={const.COLUMN_0: const.TIMESTAMP_COLUMN}
            )

            # Adjust types of the columns
            adjusted_df = self._adjust_types(melted_single_measure_df, variable=variable)

            # Assign the measure to the correct dictionary
            # It cannot happen here that the measure time is not 1h or 24h
            if measure_time == const.TIME_1H:
                time_1h[sub_dir].append(adjusted_df)
            elif measure_time == const.TIME_24H:
                time_24h[sub_dir].append(adjusted_df)

        # Concatenate the dictionaries into DataFrames
        df_time_1h = self._concat_df(time_1h)
        df_time_24h = self._concat_df(time_24h)

        # Resample the DataFrames to the specified measurement time
        # Ensure that all dates are present in the DataFrame
        df_time_1h = self._resample_df(df_time_1h, const.TIME_1H)
        df_time_24h = self._resample_df(df_time_24h, const.TIME_24H)

        return df_time_1h, df_time_24h

    def _extract_measurement_metadata(self, file_name: str) -> Tuple[Optional[MeasureTime], str]:
        """Scrap the measurement metadata from the DataFrame.

        Parameters
        ----------
        file_name : str
            The name of the file.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the measurement time and the variable name.
        """
        # Remove .xslx extension from the file name
        file_parts = file_name.split(".")[0]

        # Split the file name by "_", as the files are <year>_<variable>_<measure_time>.xlsx
        # E.g. "2018_Jony_PM25_24g.xlsx"
        file_parts = file_parts.split("_")

        # Extract the measure time from the file name
        measure_time = file_parts[-1]

        # Check measure time
        if measure_time != const.TIME_1H and measure_time != const.TIME_24H:
            measure_time = None

        # Some variable names are at least two parts name separated by "_"
        # E.g. "2018_Jony_PM25_24g.xlsx"
        variable = "_".join(file_parts[1:-1])

        # Rename the the PM25 to PM2.5
        variable = const.PM2_5 if variable == const.PM25 else variable

        return measure_time, variable

    def _scrap_sensor_names(
        self, data: pd.DataFrame, first_date_index: int, file_name: str
    ) -> Dict[str, str]:
        """Scrap the sensor names from the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the measurement data.
        first_date_index : int
            The index of the first date in the DataFrame.
        file_name : str
            The name of the file.

        Returns
        -------
        List[str]
            A list of sensor names.
        """
        try:
            # Try to extract the sensor names based on the KOD STACJI row
            sensor_names = data.query(f"`{const.COLUMN_0}` == '{const.KOD_STACJI}'").iloc[0, 1:]
            return sensor_names.to_dict()

        except IndexError:
            logging.info(
                "No sensor names found in file %s. Trying to extract in another way.",
                file_name,
            )
            # Select first three values from the non-date rows
            df = data.iloc[:first_date_index, 1 : const.KOD_STACJI_INFERRING_ROW_SAMPLE + 1]
            for index, row in df.iterrows():
                # Process each of the first N columns
                for value in row:
                    # Check if the value is identified as old station code
                    old_station_mask = self.sensor_metadata[const.OLD_STATION_CODE].apply(
                        lambda x: value in x
                    )
                    if old_station_mask.any():
                        logging.info("Found row with station codes")
                        return data.iloc[index, 1:].to_dict()  # type: ignore

                    # Check if the value is identified as station code
                    station_mask = self.sensor_metadata[const.STATION_CODE] == value
                    if station_mask.any():
                        logging.info("Found row with station codes")
                        return data.iloc[index, 1:].to_dict()  # type: ignore

            # Case, where the sensor names are not found in the file
            logging.warning(
                "No sensor names found in file %s."
                " Manually investigate the file or try to increase"
                " the const.KOD_STACJI_INFERRING_ROW_SAMPLE"
                " Skipping.",
                file_name,
            )
            return {}

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
        # Looking for the first datetime value in the column
        for index, value in enumerate(data[const.COLUMN_0]):
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
            old_station_code_mask = self.sensor_metadata[const.OLD_STATION_CODE].apply(
                lambda x: value in x
            )
            old_sensor_names_df = self.sensor_metadata[old_station_code_mask]
            if not old_sensor_names_df.empty:
                station_codes[key] = old_sensor_names_df[const.SENSOR_ID].tolist()
                station_codes[key] = [int(x) for x in station_codes[key]]
                continue

            # If the value is not present in the old station code column,
            # use the station code to find the sensor ID
            # Check if the value is present in the station code column
            station_code_mask = self.sensor_metadata[const.STATION_CODE] == value
            sensor_names_df = self.sensor_metadata[station_code_mask]
            if sensor_names_df.empty:
                logging.warning("Sensor name %s not found. Skipping.", value)
                continue
            if len(sensor_names_df) > 1:
                logging.warning("Multiple sensor names found for %s. Skipping.", value)
                continue
            station_codes[key] = sensor_names_df[const.SENSOR_ID].tolist()
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
        cols_to_keep = [const.COLUMN_0] + list(sensor_codes.keys())

        # Select the columns to duplicate - identified sensor codes with multiple IDs
        columns_to_duplicate = [key for key, value in sensor_codes.items() if len(value) > 1]

        # The columns to rename are the columns that are not duplicated
        col_to_rename = list(set(cols_to_keep) - set(columns_to_duplicate) - {const.COLUMN_0})

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
        df_with_duplicated = df.loc[:, const.COLUMN_0].to_frame()
        for col in columns_to_duplicate:
            new_columns = sensor_codes[col]
            for new_col in new_columns:
                df_with_duplicated[new_col] = df[col].copy()

        # Concatenate the DataFrames
        to_remove_duplicated = pd.concat(
            [df_with_duplicated, df_renamed], axis=1, ignore_index=False
        )

        # Duplicated column names
        duplicated_columns_mask = to_remove_duplicated.columns.duplicated(keep=False)
        duplicated_columns = to_remove_duplicated.columns[duplicated_columns_mask].unique()

        # Remove the duplicated columns from the DataFrame
        result = to_remove_duplicated.loc[:, ~duplicated_columns_mask]

        # Combine first the duplicated columns
        for col in duplicated_columns:
            subset_with_duplicates: pd.DataFrame = to_remove_duplicated.loc[:, col]  # type: ignore
            for i in range(1, len(subset_with_duplicates.columns)):
                subset_with_duplicates.iloc[:, 0] = subset_with_duplicates.iloc[:, 0].combine_first(
                    subset_with_duplicates.iloc[:, i]
                )
            result = pd.concat([result, subset_with_duplicates.iloc[:, 0].to_frame()], axis=1)

        return result

    def _adjust_types(self, data: pd.DataFrame, variable: str) -> pd.DataFrame:
        """Adjust the types of the DataFrame columns.
        Convert the TIMESTAMP_COLUMN to datetime.
        Convert the UNIQUE_ID column to int.
        Convert variable columns to float.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the measurement data.
        variable : str
            The name of the variable column.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the adjusted types.
        """
        df = data.copy()
        df[const.TIMESTAMP_COLUMN] = pd.to_datetime(
            df[const.TIMESTAMP_COLUMN],
            format=const.DATE_FORMAT,
        )
        df[const.UNIQUE_ID] = df[const.UNIQUE_ID].astype(int)

        # Some data is reported as float some as string
        df[variable] = df[variable].astype(str)
        df[variable] = df[variable].str.replace(",", ".")
        df[variable] = df[variable].astype(float)

        return df

    def _concat_df(self, dict_of_data: Dict[str, list]) -> pd.DataFrame:
        """Concatenate the DataFrames in the list.

        Parameters
        ----------
        dict_of_data : Dict[str, list]
            The dictionary containing the DataFrames to concatenate.
            The key is the subdirectory name and the value is the list of DataFrames.

        Returns
        -------
        pd.DataFrame
            The concatenated DataFrame.
        """

        def merge_func(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
            """Heleper function to merge two DataFrames."""
            return pd.merge(x, y, how="outer", on=[const.TIMESTAMP_COLUMN, const.UNIQUE_ID])

        df = pd.concat(
            [reduce(merge_func, dict_of_data[key]) for key in dict_of_data.keys()],
            ignore_index=True,
        )

        return df

    def _resample_df(self, data: pd.DataFrame, measure_time: MeasureTime) -> pd.DataFrame:
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
        min_date = data[const.TIMESTAMP_COLUMN].min()
        max_date = data[const.TIMESTAMP_COLUMN].max()
        result = []
        freq = "h" if measure_time == const.TIME_1H else "24h"
        for unique_id in data[const.UNIQUE_ID].unique():
            # Select the subset of the DataFrame for the unique ID
            df_id = data.query(f"`{const.UNIQUE_ID}` == {unique_id}")

            # If not min_date or max_date in the DataFrame, add them
            if min_date not in df_id[const.TIMESTAMP_COLUMN].unique():
                row = pd.DataFrame(
                    {
                        const.TIMESTAMP_COLUMN: [min_date],
                        const.UNIQUE_ID: [unique_id],
                    }
                )
                df_id = pd.concat([row, df_id], ignore_index=True)
            if max_date not in df_id[const.TIMESTAMP_COLUMN].unique():
                row = pd.DataFrame(
                    {
                        const.TIMESTAMP_COLUMN: [max_date],
                        const.UNIQUE_ID: [unique_id],
                    }
                )
                df_id = pd.concat([df_id, row], ignore_index=True)

            # Sort the DataFrame by the timestamp column
            df_id = df_id.sort_values(by=[const.TIMESTAMP_COLUMN], ascending=True)

            # Resample the DataFrame to the specified frequency
            df_id = df_id.set_index(const.TIMESTAMP_COLUMN, drop=True)
            df_id = df_id.asfreq(freq)

            # Assign the unique ID to the DataFrame to avoid nans in the unique ID column
            df_id[const.UNIQUE_ID] = unique_id

            # Reset the index to keep the timestamp column
            df_id = df_id.reset_index()

            result.append(df_id)
        return pd.concat(result, ignore_index=True)
