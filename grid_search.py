import json
import logging
import os
import sys
import shutil
import argparse
import time

import numpy as np
import urllib3
from influxdb import InfluxDBClient

from classes.artificial_features import ArtificialFeatures
from classes.features_analyzer import FeaturesAnalyzer
from classes.inputs_gatherer import InputsGatherer
from classes.model_trainer import ModelTrainer
from classes.grid_searcher import GridSearcher

urllib3.disable_warnings()

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

    cfg = json.loads(open(args.c).read())

    # Load the connections parameters and update the config dict with the related values
    cfg_conns = json.loads(open(cfg['connectionsFile']).read())
    cfg.update(cfg_conns)

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

    logger.info('Connection to InfluxDb server on socket [%s:%s]' % (cfg['influxDB']['host'], cfg['influxDB']['port']))
    try:
        influx_client = InfluxDBClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                       password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                       database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    # --------------------------------------------------------------------------- #
    # Functions
    # --------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------- #
    # Start calculations
    # --------------------------------------------------------------------------- #

    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    IG = InputsGatherer(influx_client, forecast_type, cfg, logger, AF)
    FA = FeaturesAnalyzer(IG, forecast_type, cfg, logger)
    MT = ModelTrainer(FA, IG, forecast_type, cfg, logger)
    GS = GridSearcher(FA, IG, MT, forecast_type, cfg, logger)

    # for dataset in cfg['datasetSettings']['customJSONSignals']:
    #     assert os.path.isfile(cfg['datasetSettings']['loadSignalsFolder'] + dataset['filename'])

    start_time = time.time()

    # Cycle over the regions
    for k_region in cfg['regions'].keys():

        FA.dataset_reader(target_column=cfg['regions'][k_region]['targetColumn'])

        GS.search_weights()

    logger.info("--- %s seconds elapsed for grid searching ---" % (time.time() - start_time))

    logger.info('Ending program')
