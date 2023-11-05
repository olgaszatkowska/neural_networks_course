import pandas as pd
from pandas.core.series import Series
import scipy.io as sio

from preprocess.paths import (
    FOREST_FIRES_RAW_FILE,
    FOREST_FIRES_PROCESSED_FILE_X,
    FOREST_FIRES_PROCESSED_FILE_y,
)

WEEKDAYS_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
MONTHS_MAP = {
    "jan": 0,
    "feb": 1,
    "mar": 2,
    "apr": 3,
    "may": 4,
    "jun": 5,
    "jul": 6,
    "aug": 7,
    "sep": 8,
    "oct": 9,
    "nov": 10,
    "dec": 11,
}


def _process_forest_element(val: str | float) -> str | float:
    if not isinstance(val, str):
        return val

    if val in list(WEEKDAYS_MAP.keys()):
        return WEEKDAYS_MAP[val]

    if val in list(MONTHS_MAP.keys()):
        return MONTHS_MAP[val]

    raise Exception("Value not matched")


def _process_forest_row(series: Series) -> Series:
    return series.map(_process_forest_element)


def generate_forest_fires_files() -> list[str]:
    area_header = "area"
    df = pd.read_csv(FOREST_FIRES_RAW_FILE)
    df = df.apply(_process_forest_row)
    df = df.astype(float)

    df_y = df[[area_header]]
    # Normalize the outputs
    df_y = df_y / df_y.max()
    df_y.to_csv(FOREST_FIRES_PROCESSED_FILE_y, header=None, index=False)

    df_X = df
    df_X.drop(area_header, axis=1, inplace=True)

    df_X.to_csv(FOREST_FIRES_PROCESSED_FILE_X, header=None, index=False)
    return FOREST_FIRES_PROCESSED_FILE_X, FOREST_FIRES_PROCESSED_FILE_y
