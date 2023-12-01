import pandas as pd
from pandas import DataFrame


def main():
    name = 'two_dataset'

    dataframe: DataFrame = pd.read_csv(
        f'C:/Users/egork/Desktop/Сlassifier/data/{name}.csv'
    )

    dataframe.to_csv(f'C:/Users/egork/Desktop/Сlassifier/data/super_optimize_{name}.gz', index=False)
    dataframe.to_csv(f'C:/Users/egork/Desktop/Сlassifier/data/super_optimize_{name}.bz2', index=False)
    dataframe.to_csv(f'C:/Users/egork/Desktop/Сlassifier/data/super_optimize_{name}.zip', index=False)
    dataframe.to_csv(f'C:/Users/egork/Desktop/Сlassifier/data/super_optimize_{name}.xz', index=False)


if __name__ == '__main__':
    main()
