import numpy as np
import pandas as pd


def main():
    df = pd.read_csv(r'C:/Users/egork/Desktop/Сlassifier/data/optimize_dataset.csv')

    for i in range(5):
        df = df.reindex(np.random.permutation(df.index))

    df.to_csv(r'C:/Users/egork/Desktop/Сlassifier/data/permutation_dataset.csv', index=False)


if __name__ == '__main__':
    main()
