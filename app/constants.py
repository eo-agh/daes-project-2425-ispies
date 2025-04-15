"""Module containing constants for the application."""

# Sensor Ingestor constants
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

# Measurement Ingestor constants
COLUMN_0 = "0"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

KOD_STACJI = "Kod stacji"
CZAS_USREDNIANIA = "Czas uśredniania"
WSKAZNIK = "Wskaźnik"

KOD_STACJI_INFERRING_ROW_SAMPLE = 3

DEPOZYCJA = "depozycja"

TIME_1H = "1g"
TIME_24H = "24g"

TIMESTAMP_COLUMN = "ds"
UNIQUE_ID = "unique_id"

C6H6 = "C6H6"
CO = "CO"
NO = "NO"
NO2 = "NO2"
NOX = "NOx"
O3 = "O3"
PM25 = "PM25"
PM2_5 = "PM2.5"
PM10 = "PM10"
SO2 = "SO2"

TARGET_VARIABLES = (C6H6, CO, NO, NO2, NOX, O3, PM2_5, PM10, SO2)
