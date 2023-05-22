import json
import logging
import os
import sys
import argparse

from classes.artificial_features import ArtificialFeatures
from classes.features_analyzer import FeaturesAnalyzer
from classes.inputs_gatherer import InputsGatherer

from classes.optimized_model_creator import OptimizedModelCreator

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", help="configuration file")
    arg_parser.add_argument("-t", help="type (MOR | EVE)")
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

    # Define the forecast type
    forecast_type = args.t

    # Set logging object
    if not args.l:
        log_file = None
    else:
        log_file = args.l

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=log_file)

    logger.info('Starting program')

    af = ArtificialFeatures(None, forecast_type, cfg, logger)
    ig = InputsGatherer(None, forecast_type, cfg, logger, af)

    procs = []
    # Cycle over the regions
    for k_region in cfg['regions'].keys():
        for target in cfg['regions'][k_region]['targets']:

            # Phase N°1: Data retrieving
            fa = FeaturesAnalyzer(ig, forecast_type, cfg, logger)
            fa.dataset_reader(k_region, [target])
            dataset = fa.dataFrames[k_region]['dataset']
            root_folder = fa.inputs_gatherer.output_folder_creator(k_region)

            logger.info('Dataset main settings: observations = %i, features = %i' % (len(dataset), len(dataset.columns)))
            omc = OptimizedModelCreator(dataset, target, k_region, forecast_type, root_folder, cfg, logger)

            # Phase N°2: First (eventual) hyperparameters optimization, performed considering all the features
            if cfg['hpoBeforeFS']['enabled'] is True:
                logger.info('First HPOPT starting')
                omc.do_hyperparameters_optimization('before_fs')
                logger.info('First HPOPT ending')

            # Phase N°3: Features selection via Shapley values considering the optimized hyperparameters
            logger.info('FS starting')
            omc.do_feature_selection()
            logger.info('FS ending')

            # Phase N°4: Second hyperparameters optimization, performed considering only the features selected by FS
            logger.info('Second HPOPT starting')
            omc.do_hyperparameters_optimization('after_fs')
            logger.info('Second HPOPT ending')

            # # Phase N°5: Model training
            logger.info('MT starting')
            omc.do_models_training()
            logger.info('MT ending')

    logger.info('Ending program')
