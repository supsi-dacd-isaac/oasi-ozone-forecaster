import argparse
import logging
import numpy as np
import pandas as pd

# Example:
# python dataset_cleaner.py --input_file output_pm10/SPC_MOR_20161001_20220401/SPC_MOR_20161001_20220401_dataset.csv --signal GIU__WSgust,GIU__Gl,TIGIU__GLOB,TIGIU__GLOB_c2
# python dataset_cleaner.py --input_file output_pm10/STC_MOR_20161001_20220401/STC_MOR_20161001_20220401_dataset.csv --signal BIO__Gl,P_BIO__GLOB,P_BIO__GLOB_c2,BIO__RH,P_BIO__RELHUM_2M,P_BIO__RELHUM_2M_c2

def save_output_file(df, f):
    # Saved the joined dataset
    logger.info('START WRITE ON %s]' % f)
    df.to_csv(f, mode='w', header=True, index=False)
    logger.info('END WRITE ON %s]' % f)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_file', required=True)
    arg_parser.add_argument('--signals', required=True)
    args = arg_parser.parse_args()

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=None)

    # Main variables initialization
    input_file = args.input_file
    signals = args.signals.split(',')

    logger.info('STARTING PROGRAM')
    # Read the dataset
    logger.info('START INPUT FILES READING')
    df = pd.read_csv(input_file)
    logger.info('END INPUT FILES READING')

    cols_to_drop = []
    for signal in signals:
        logger.info('DROP COLUMNS RELATED TO %s' % signal)
        for col in df.columns:
            if signal in col:
                cols_to_drop.append(col)
    output_file = input_file.replace('.csv', '_cleaned.csv')
    df = df.drop(columns=cols_to_drop)

    logger.info('SAVE UPDATED DATAFRAME ON FILE %s' % output_file)
    df.to_csv(output_file, mode='w', header=True, index=False)

    logger.info('ENDING PROGRAM')

