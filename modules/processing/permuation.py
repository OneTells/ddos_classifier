import numpy as np
import pandas as pd


def main():
    df = pd.read_csv(r'C:\Users\egork\Desktop\DDoS\data\optimize_dataset.csv')

    for i in range(10):
        df = df.reindex(np.random.permutation(df.index))

    df.to_csv(r'C:\Users\egork\Desktop\DDoS\data\permutation_dataset.csv', index=False)


if __name__ == '__main__':
    main()
