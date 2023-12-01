import os
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader

from modules.learning.action import Action


class ClassifierEnv(Env[np.ndarray, int]):

    def __init__(self, render_mode: str = None):
        self.render_mode = render_mode

        self.__dataframe: TextFileReader = self.__get_dataset()

        self.action_space = Discrete(11)

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

        self.__actions: tuple[Action, ...] = (
            Action(0, 0, 'mptf'),
            Action(0, 0, 'mbtf'),
            Action(0, 0, 'mprf'),
            Action(0, 0, 'mbrf'),
            Action(0, 0, 'rtp')
        )

    @staticmethod
    def __get_dataset() -> TextFileReader:
        return pd.read_csv(f'{os.getcwd()}/data/super_optimize_one_dataset.bz2', chunksize=1000)

    def __check_row(self, row: Series) -> bool:
        for action in self.__actions:
            if action.check(row):
                return True

        return False

    def __reward(self, dataframe: DataFrame) -> float:
        successful_classifications_number = 0

        for row in dataframe:
            if self.__check_row(row) == bool(row['label']):
                successful_classifications_number += 1

        return float(round(successful_classifications_number / len(dataframe) * 100))

    def reset(self, *_) -> tuple[np.ndarray, dict[str, Any]]:
        self.__dataframe: TextFileReader = self.__get_dataset()
        return np.array([action.reset() for action in self.__actions], dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        try:
            dataframe = next(self.__dataframe)
        except StopIteration:
            return np.array([action.threshold for action in self.__actions], dtype=np.float32), 0, True, False, {}

        if action != 10:
            if action % 2 == 0:
                self.__actions[action // 2].increase()
            else:
                self.__actions[action // 2].decrease()

        reward = self.__reward(dataframe)
        return np.array([action.threshold for action in self.__actions], dtype=np.float32), reward, False, False, {}
