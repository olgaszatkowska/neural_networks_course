import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

from dataset.preprocess.forest_fires import generate_forest_fires_files
from dataset.preprocess.affnist import load_affnist


class Dataset:
    def __init__(self) -> None:
        self._forest_fires_files = generate_forest_fires_files()

    def forest_fires_dataset(self) -> list[NDArray]:
        return self._get_data_set(*self._forest_fires_files)

    def affnist_dataset(self):
        return self._train_test_split(*load_affnist(page=2))

    def _train_test_split(self, df_X: NDArray, df_y: NDArray):
        return train_test_split(df_X, df_y, test_size=0.2, random_state=242)

    def _get_data_set(self, path_X: str, path_y: str) -> list[NDArray]:
        np.set_printoptions(suppress=True)
        df_X = pd.read_csv(path_X).to_numpy()
        df_y = pd.read_csv(path_y).to_numpy().reshape(-1)
        return self._train_test_split(df_X, df_y)
