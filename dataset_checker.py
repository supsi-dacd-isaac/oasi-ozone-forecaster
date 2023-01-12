import argparse
import logging
import pandas as pd

# Example:
# python dataset_checker.py -i output_pm10/SPC_MOR_20211201_20211205/SPC_MOR_20211201_20211205_dataset.csv
# python dataset_checker.py -i output_pm10/SPC_MOR_20211201_20211205/SPC_MOR_20211201_20211205_dataset.csv -v

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', required=True)
    arg_parser.add_argument('-v', action='store_const', const=True, default=False)
    args = arg_parser.parse_args()

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=None)

    input = args.i

    df = pd.read_csv(input)
    df = df.set_index('date')

    print('\nSTARTING PROGRAM')

    cnt_nans_cases = 1
    tot_sig_code = dict()
    for k in df.columns:
        cnt_nans = df[k].isna().sum()
        if cnt_nans > 0:
            tmp = k.split('__')
            sig_code = '%s__%s' % (tmp[0], tmp[1])
            if sig_code not in tot_sig_code.keys():
                tot_sig_code[sig_code] = [cnt_nans]
            else:
                tot_sig_code[sig_code][0] += cnt_nans

            cnt_nans_cases += 1

    if args.v is True:
        print('\nDAYS WITH NANS:')
        days_with_nans = df[df.isna().any(axis=1)].index
        for day in days_with_nans:
            print(day)

        print('\nSIGNALS WITH NANS:')
        for k in tot_sig_code.keys():
            if tot_sig_code[k][0] > 5:
                print('%s = %i' % (k, tot_sig_code[k][0]))

    print('\nFIRST DAY: %s' % df.head(1).index[0])
    print('LAST DAY: %s' % df.tail(1).index[0])
    print('INPUT SIGNALS WITHOUT NANS: %i/%i' % (len(df.columns)-cnt_nans_cases, len(df.columns)))
    print('DAYS WITHOUT NANS: %i/%i (%.0f%%)' % (len(df.dropna()), len(df), len(df.dropna())*1e2/len(df)))

    print('\nENDING PROGRAM\n')
