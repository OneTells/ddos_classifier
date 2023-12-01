from pandas import Series


class Action:

    def __init__(self, threshold: float, step: float, key: str) -> None:
        self.threshold: float = threshold
        self.__default_threshold: float = threshold

        self.__step: float = step
        self.__key: str = key

    def check(self, row: Series) -> bool:
        return row[self.__key] > self.threshold

    def increase(self) -> None:
        self.threshold += self.__step

    def decrease(self) -> None:
        self.threshold -= self.__step

    def reset(self) -> float:
        self.threshold: float = self.__default_threshold
        return self.threshold
