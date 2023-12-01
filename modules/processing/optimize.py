import os

import pandas as pd
from pandas import DataFrame


def main():
    dataframe: DataFrame = pd.read_csv(
        r'C:/Users/egork/Desktop/Сlassifier/data/new_dataset.csv'
    )

    dataframe['mptf'] = dataframe['mptf'].apply(int)
    dataframe['mbtf'] = dataframe['mbtf'].apply(int)
    dataframe['mprf'] = dataframe['mprf'].apply(int)
    dataframe['mbrf'] = dataframe['mbrf'].apply(int)
    dataframe['label'] = dataframe['label'].apply(int)

    dataframe.to_csv(r'C:/Users/egork/Desktop/Сlassifier/data/optimize_dataset.csv', index=False)


if __name__ == '__main__':
    main()
