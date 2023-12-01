import pandas as pd


def main():
    df = pd.read_csv(r'C:/Users/egork/Desktop/Сlassifier/data/permutation_dataset.csv')

    len_ = len(df)

    df.loc[0:int(len_ * 0.8)].to_csv(r'C:/Users/egork/Desktop/Сlassifier/data/one_dataset.csv', index=False)
    df.loc[int(len_ * 0.8) + 1:].to_csv(r'C:/Users/egork/Desktop/Сlassifier/data/two_dataset.csv', index=False)


if __name__ == '__main__':
    main()
