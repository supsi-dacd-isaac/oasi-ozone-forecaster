import json
import logging
import os
import sys
import argparse

from classes.inputs_gatherer import InputsGatherer
from classes.optimized_model_creator import OptimizedModelCreator

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
    for k_region in cfg['regions'].keys():
        for forecast_type in cfg['regions'][k_region]['targets'].keys():
            ig = InputsGatherer(None, forecast_type, cfg, logger, None)

            for target in cfg['regions'][k_region]['targets'][forecast_type]:

                ft_desc = '%s__%s -> ' % (forecast_type, target)
                # Phase N°1: Data retrieving
                omc = OptimizedModelCreator(ig, target, k_region, forecast_type, cfg, logger)
                omc.fill_datasets(k_region, target)

                logger.info('%sDataset dimensions: observations = %i, features = %i' % (ft_desc, len(omc.dataset),
                                                                                        len(omc.dataset.columns)-1))

                # Phase N°2: First (eventual) hyperparameters optimization, performed considering all the features
                if cfg['hpoBeforeFS']['enabled'] is True:
                    logger.info('%sFirst HPOPT starting' % ft_desc)
                    omc.do_hyperparameters_optimization('before_fs')
                    logger.info('%sFirst HPOPT ending' % ft_desc)

                # Phase N°3: Features selection via Shapley values considering the optimized hyperparameters
                logger.info('%sFS starting' % ft_desc)
                omc.do_feature_selection()
                logger.info('%sFS ending' % ft_desc)

                # Phase N°4: Features linear correlation analysis
                if cfg['fs']['corrAnalysis']['enabled'] is True:
                    logger.info('%sFCA starting' % ft_desc)
                    omc.corr_analysis(target)
                    logger.info('%sFCA ending' % ft_desc)

                # Phase N°5: Second hyperparameters optimization, performed considering only the features selected by FS
                logger.info('%sSecond HPOPT starting' % ft_desc)
                omc.do_hyperparameters_optimization('after_fs')
                logger.info('%sSecond HPOPT ending' % ft_desc)

                # Phase N°6: Model training
                logger.info('%sMT starting' % ft_desc)
                omc.do_models_training()
                logger.info('%sMT ending' % ft_desc)

    logger.info('Ending program')
