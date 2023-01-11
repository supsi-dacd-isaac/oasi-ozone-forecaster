import argparse
import logging
import numpy as np
import pandas as pd

# Example:
# 1) Join by columns
# python ../dataset_joiner.py --if1 SPC_EVE_20150515_20210915_dataset_f1.csv --if2 SPC_EVE_20150515_20210915_dataset_f2.csv --of SPC_EVE_20150515_20210915_dataset.csv --join colums --y YO3
# 2) Join by rows
# python ../dataset_joiner.py --if1 tmp/if1.csv --if2 tmp/if2.csv --of tmp/of.csv --join rows

def save_output_file(df, f):
    # Saved the joined dataset
    logger.info('START WRITE ON %s]' % f)
    df.to_csv(f, mode='w', header=True, index=False)
    logger.info('END WRITE ON %s]' % f)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--if1', required=True)
    arg_parser.add_argument('--if2', required=True)
    arg_parser.add_argument('--of', required=True)
    arg_parser.add_argument('--join', required=True)
    arg_parser.add_argument('--y', required=False)
    args = arg_parser.parse_args()

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=None)

    # Main variables initialization
    if1 = args.if1
    if2 = args.if2
    of = args.of
    join_type = args.join

    logger.info('STARTING PROGRAM')
    # Read the dataset
    logger.info('START INPUT FILES READING')
    df1 = pd.read_csv(if1)
    df2 = pd.read_csv(if2)
    logger.info('END INPUT FILES READING')

    if join_type == 'columns':
        # Drop the targets columns from df1
        logger.info('DATASETS INITIALIZATION')
        y = args.y
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

        # Save the output dataframe
        save_output_file(dfo, of)
    elif join_type == 'rows':
        if np.array_equiv(np.array(df1.columns), np.array(df2.columns)) is True:
            dfo = pd.concat([df1, df2])

            # Save the output dataframe
            save_output_file(dfo, of)
        else:
            logger.warning('Mismatch in the columns dataframe, impossible to merge the data')
    else:
        logger.warning('Join type must be \'rows\' or \'columns\'')

    logger.info('ENDING PROGRAM')

