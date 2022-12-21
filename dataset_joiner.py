import argparse
import logging
import pandas as pd

# Example:
# python ../dataset_joiner.py --if1 SPC_EVE_20150515_20210915_dataset_f1.csv --if2 SPC_EVE_20150515_20210915_dataset_f2.csv --of SPC_EVE_20150515_20210915_dataset.csv --y YO3

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--if1', required=True)
    arg_parser.add_argument('--if2', required=True)
    arg_parser.add_argument('--of', required=True)
    arg_parser.add_argument('--y', required=True)
    args = arg_parser.parse_args()

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=None)

    # Main variables initialization
    if1 = args.if1
    if2 = args.if2
    of = args.of
    y = args.y

    logger.info('STARTING PROGRAM')

    # Read the dataset
    logger.info('START INPUT FILES READING')
    df1 = pd.read_csv(if1)
    df2 = pd.read_csv(if2)
    logger.info('END INPUT FILES READING')

    # Drop the targets columns from df1
    logger.info('DATASETS INITIALIZATION')
    cols_to_drop = []
    for col in df1.columns:
        if y in col:
            cols_to_drop.append(col)
    df1 = df1.drop(columns=cols_to_drop)

    # Drop the date column from df1
    df2 = df2.drop(columns=['date'])

    # Join the datasets
    logger.info('START JOIN[%s:%s]' % (if1, if2))
    dfo = pd.concat([df1, df2], axis=1, join='inner')
    logger.info('END JOIN[%s:%s]' % (if1, if2))

    # Saved the joined dataset
    logger.info('START WRITE ON %s]' % of)
    dfo.to_csv(of, mode='w', header=True, index=False)
    logger.info('END WRITE ON %s]' % of)

    logger.info('ENDING PROGRAM')

