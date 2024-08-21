import json
import logging
import os
import sys
import argparse
import pandas as pd

from classes.optimized_model_creator_v2 import OptimizedModelCreatorV2


def get_data_file_path(region, main_cfg):
    str_input_folder = '%s%s_%s%s_%s%s%s' % (main_cfg['outputFolder'], region,
                                             main_cfg['dataset']['startYear'],
                                             main_cfg['dataset']['startDay'],
                                             main_cfg['dataset']['endYear'],
                                             main_cfg['dataset']['endDay'], os.sep)
    return '%s%s_dataset.csv' % (str_input_folder, str_input_folder.split(os.sep)[1])


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", help="configuration file")
    arg_parser.add_argument("-l", help="log file (optional, if empty log redirected on stdout)")
    args = arg_parser.parse_args()

    # Load the main parameters
    config_file = args.c
    if os.path.isfile(config_file) is False:
        print('\nATTENTION! Unable to open configuration file %s\n' % config_file)
        sys.exit(1)

    # Load the configuration
    cfg_file_name = args.c
    cfg = json.loads(open(cfg_file_name).read())

    cfg_signals_codes = json.loads(open(cfg['signalsCodesFile']).read())
    cfg.update(cfg_signals_codes)

    # Set logging object
    if not args.l:
        log_file = None
    else:
        log_file = args.l

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=log_file)

    logger.info('Starting program')

    # Cycle over the regions
    for k_location in cfg['locations'].keys():
        for predictor_cfg in cfg['locations'][k_location]['predictors']:
            omc_v2 = OptimizedModelCreatorV2(k_location, predictor_cfg, cfg, logger)

            # Raw data loading
            raw_df = pd.read_csv(get_data_file_path(cfg['locations'][k_location]['region'], cfg), index_col=0, parse_dates=True)
            # raw_df = raw_df.head(2000)

            # Dataset creation
            omc_v2.io_dataset_creation(raw_df)

            # Feature selection
            omc_v2.fs()

            # Optimization
            omc_v2.optimize()

            # Training
            omc_v2.train()

            # Data saving
            omc_v2.save()

    logger.info('Ending program')
