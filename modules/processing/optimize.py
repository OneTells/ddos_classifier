import pandas as pd
from pandas import DataFrame


def main():
    dataframe: DataFrame = pd.read_csv(
        'C:/Users/egork/Desktop/DDoS/data/new_dataset.csv'
    )

    dataframe['mptf'] = dataframe['mptf'].apply(int)
    dataframe['mbtf'] = dataframe['mbtf'].apply(int)
    dataframe['mprf'] = dataframe['mprf'].apply(int)
    dataframe['mbrf'] = dataframe['mbrf'].apply(int)

    dataframe.to_csv('C:/Users/egork/Desktop/DDoS/data/optimize_dataset.csv', index=False)


if __name__ == '__main__':
    main()
