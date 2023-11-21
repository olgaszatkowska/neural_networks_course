import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

from dataset.preprocess.forest_fires import generate_forest_fires_files
from dataset.preprocess.affnist import load_affnist


def load_forest_fires_dataset() -> list[NDArray]:
    path_X, path_y = generate_forest_fires_files()
    np.set_printoptions(suppress=True)
    df_X = pd.read_csv(path_X).to_numpy()
    df_y = pd.read_csv(path_y).to_numpy().reshape(-1)

    return _train_test_split(df_X, df_y)


def load_affnist_dataset() -> list[NDArray]:
    return _train_test_split(*load_affnist(page=2))


def load_fashion_mnist_dataset(noisy: bool = False, noise_level: float = 0.1, y: bool = False) -> list[NDArray]:
    from tensorflow.keras.datasets import fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    if y:
        return x_train.reshape(-1, 784), y_train, x_test.reshape(-1, 784), y_test
    
    if noisy:
        noise = noise_level * np.random.randn(*x_train.shape)
        noisy_x_train = x_train + noise
        
        return noisy_x_train.reshape(-1, 784), x_train.reshape(-1, 784)

    return x_train.reshape(-1, 784), x_test.reshape(-1, 784)


def _train_test_split(df_X: NDArray, df_y: NDArray):
    return train_test_split(df_X, df_y, test_size=0.2, random_state=242)
