import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

from preprocess.forest_fires import generate_forest_fires_files
from preprocess.affnist import load_affnist


class Dataset:
    def __init__(self) -> None:
        self._forest_fires_files = generate_forest_fires_files()

    def _get_data_set(self, path_X: str, path_y: str) -> list[NDArray]:
        df_X = pd.read_csv(path_X)
        df_y = pd.read_csv(path_y)
        return train_test_split(df_X, df_y, test_size=0.2, random_state=242)

    def forest_fires_dataset(self) -> list[NDArray]:
        return self._get_data_set(*self._forest_fires_files)

    def affnist_dataset(self):
        return load_affnist(page=2)
