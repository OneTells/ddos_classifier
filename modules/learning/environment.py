from typing import Any

import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete, Box
from numpy import ndarray, dtype
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader

from modules.learning.filter import Filter


class ClassifierEnv:

    def __init__(self, dataset_path: str, dataset_count_row: int = 1000):
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
        self.filters: tuple[Filter, ...] = (
            Filter(3, 1, 'mptf'),
            Filter(85, 5, 'mbtf'),
            Filter(4, 1, 'mprf'),
            Filter(45, 5, 'mbrf'),
            Filter(0.13, 0.01, 'rtp')
        )

        self.report_y_true = []
        self.report_y_answer = []

    def __get_dataset(self) -> TextFileReader:
        return pd.read_csv(
            self.__dataset_path, chunksize=self.__dataset_count_row
        )

    def __check_row(self, row: Series) -> bool:
        for action in self.filters:
            if action.check(row):
                return True

        return False

    def __reward(self, dataframe: DataFrame) -> float:
        successful_classifications_number = 0

        for _, row in dataframe.iterrows():
            if (answer := self.__check_row(row)) == bool(row['label']):
                successful_classifications_number += 1

            self.report_y_true.append(bool(row['label']))
            self.report_y_answer.append(answer)

        if round(successful_classifications_number / len(dataframe) * 100) >= 50:
            return 1

        return -1

    def reset(self, *_) -> ndarray[Any, dtype[Any]]:
        self.__dataframe: TextFileReader = self.__get_dataset()
        return np.array([filter_.reset() for filter_ in self.filters], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        try:
            dataframe = next(self.__dataframe)
        except StopIteration:
            return np.array([action.threshold for action in self.filters], dtype=np.float32), 0, True

        if action != 10:
            if action % 2 == 0:
                self.filters[action // 2].increase()
            else:
                self.filters[action // 2].decrease()

        reward = self.__reward(dataframe)
        return np.array([filter_.threshold for filter_ in self.filters], dtype=np.float32), reward, False
