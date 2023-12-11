import os
from typing import Any

import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete, Box
from numpy import ndarray, dtype, floating
from numpy._typing import _32Bit
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader

from modules.learning.filter import Filter


class ClassifierEnv:

    def __init__(self, dataset_path: str = None, dataset_count_row: int = 1000):
        self.__dataset_path = dataset_path
        self.__dataset_count_row = dataset_count_row

        self.__dataframe: TextFileReader = self.__get_dataset()

        self.action_space: Discrete = Discrete(11)

        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.observation_space = Box(low, high, dtype=np.float32)

        self.__filters: tuple[Filter, ...] = (
            Filter(3, 1, 'mptf'),
            Filter(40, 5, 'mbtf'),
            Filter(3, 1, 'mprf'),
            Filter(50, 5, 'mbrf'),
            Filter(0.1, 0.01, 'rtp')
        )

    def __get_dataset(self) -> TextFileReader:
        return pd.read_csv(
            self.__dataset_path or f'{os.getcwd()}/data/super_optimize_two_dataset.bz2', chunksize=self.__dataset_count_row
        )

    def __check_row(self, row: Series) -> bool:
        for action in self.__filters:
            if action.check(row):
                return True

        return False

    def __reward(self, dataframe: DataFrame) -> float:
        successful_classifications_number = 0

        for _, row in dataframe.iterrows():
            if self.__check_row(row) == bool(row['label']):
                successful_classifications_number += 1

        return float(round(successful_classifications_number / len(dataframe) * 100))

    def reset(self, *_) -> ndarray[Any, dtype[Any]]:
        self.__dataframe: TextFileReader = self.__get_dataset()
        return np.array([filter_.reset() for filter_ in self.__filters], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        try:
            dataframe = next(self.__dataframe)
        except StopIteration:
            return np.array([action.threshold for action in self.__filters], dtype=np.float32), 0, True

        if action != 10:
            if action % 2 == 0:
                self.__filters[action // 2].increase()
            else:
                self.__filters[action // 2].decrease()

        reward = self.__reward(dataframe)
        return np.array([filter_.threshold for filter_ in self.__filters], dtype=np.float32), reward, False
