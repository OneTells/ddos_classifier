from csv import DictWriter
from typing import Iterable

import pandas as pd
from pandas import Series, DataFrame

label_dict = {'ddos': 1, 'Benign': 0}


def formatting_row(row: Series) -> Series:
    mptf = row['Tot Fwd Pkts']
    mbtf = row['TotLen Fwd Pkts']
    mprf = row['Tot Bwd Pkts']
    mbrf = row['TotLen Bwd Pkts']

    try:
        rtp = round(row['Tot Fwd Pkts'] / (row['Tot Fwd Pkts'] + row['Tot Bwd Pkts']), 3)
    except ZeroDivisionError:
        rtp = 0.0

    return Series(dict(
        label=label_dict[row['Label']], mptf=mptf, mbtf=mbtf, mprf=mprf, mbrf=mbrf, rtp=rtp
    ))


def main():
    with open('C:/Users/egork/Desktop/DDoS/data/new_dataset.csv', 'w', newline='') as file:
        writer = DictWriter(file, fieldnames=['mptf', 'mbtf', 'mprf', 'mbrf', 'rtp', 'label'])
        writer.writeheader()

        dataframes = pd.read_csv(
            'C:/Users/egork/Desktop/DDoS/data/original_dataset.csv', chunksize=100_000
        )
        for index, dataframe in enumerate(dataframes):
            print(index)
            writer.writerows(dataframe.apply(formatting_row, axis=1).to_dict('records'))


if __name__ == '__main__':
    main()
