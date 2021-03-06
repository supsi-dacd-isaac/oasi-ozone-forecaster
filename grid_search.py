import json
import logging
import os
import sys
import argparse

from multiprocessing import Process
import urllib3
from influxdb import InfluxDBClient

from classes.artificial_features import ArtificialFeatures
from classes.features_analyzer import FeaturesAnalyzer
from classes.inputs_gatherer import InputsGatherer
from classes.model_trainer import ModelTrainer
from classes.grid_searcher import GridSearcher

urllib3.disable_warnings()


# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def gs_process(ig, k_region, target, cfg, logger, cfg_file_name):
    fa = FeaturesAnalyzer(ig, forecast_type, cfg, logger)
    mt = ModelTrainer(fa, ig, forecast_type, cfg, logger)
    gs = GridSearcher(fa, ig, mt, forecast_type, cfg, logger)

    fa.dataset_reader(k_region, [target])
    gs.search_weights(k_region, target, cfg_file_name)


if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # Configuration file
    # --------------------------------------------------------------------------- #
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

    cfg_file_name = args.c
    cfg = json.loads(open(cfg_file_name).read())

    # Define the forecast type
    forecast_type = args.t

    # --------------------------------------------------------------------------- #
    # Set logging object
    # --------------------------------------------------------------------------- #
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
        for target in cfg['regions'][k_region]['gridSearcher']['targetColumns']:
            tmp_proc = Process(target=gs_process, args=[ig, k_region, target, cfg, logger, cfg_file_name])
            tmp_proc.start()
            procs.append(tmp_proc)
            # gs_process(ig, k_region, target, cfg, logger, cfg_file_name)

    for proc in procs:
        proc.join()

    logger.info('Ending program')
