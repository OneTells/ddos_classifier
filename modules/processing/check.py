import pandas as pd


def main():
    df = pd.read_csv(r'C:/Users/egork/Desktop/Сlassifier/data/permutation_dataset.csv')
    print(len(df))

    df = pd.read_csv(r'C:/Users/egork/Desktop/Сlassifier/data/super_optimize_one_dataset.bz2')
    print(len(df))

    df = pd.read_csv(r'C:/Users/egork/Desktop/Сlassifier/data/super_optimize_two_dataset.bz2')
    print(len(df))


if __name__ == '__main__':
    main()
