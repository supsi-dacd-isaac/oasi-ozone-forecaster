import argparse
import logging
import pandas as pd

# Example:
# python dataset_checker.py -i output_pm10/SPC_MOR_20211201_20211205/SPC_MOR_20211201_20211205_dataset.csv

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', required=True)
    args = arg_parser.parse_args()

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=None)

    input = args.i

    df1 = pd.read_csv(input)

    nans = df1.isnull().any()

    print('STARTING PROGRAM')

    cnt_nans = 1
    print('INPUT WITH NANS:')
    for i in range(0, len(nans)):
        if nans[i] == True:
            print(cnt_nans, nans.index[i], nans[i])
            cnt_nans += 1

    print('ENDING PROGRAM')

