import json
import logging
import os
import sys
import shutil
import argparse

import numpy as np
import pandas as pd
import urllib3
from influxdb import InfluxDBClient

dir_path = os.path.dirname(os.path.realpath(__file__))
path_parent = os.path.dirname(dir_path)
sys.path.insert(0, path_parent)

from classes.artificial_features import ArtificialFeatures
from classes.features_analyzer import FeaturesAnalyzer
from classes.inputs_gatherer import InputsGatherer
from classes.grid_searcher import GridSearcher
from classes.model_trainer import ModelTrainer

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
    # Test using regions
    # --------------------------------------------------------------------------- #

    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    IG = InputsGatherer(influx_client, forecast_type, cfg, logger, AF)
    FA = FeaturesAnalyzer(IG, forecast_type, cfg, logger)
    MT = ModelTrainer(FA, IG, forecast_type, cfg, logger)
    GS = GridSearcher(FA, IG, MT, forecast_type, cfg, logger)

    cfg['datasetSettings']['startDay'] = '07-10'
    cfg['datasetSettings']['endDay'] = '07-20'
    cfg['datasetSettings']['years'] = [2019]
    cfg['featuresAnalyzer']['datasetCreator'] = 'regions'
    cfg['featuresAnalyzer']['performFeatureSelection'] = True
    cfg['featuresAnalyzer']['numberSelectedFeatures'] = 10

    # FA.dataset_creator()
    FA.dataset_reader()

    MT.train_final_models()

    for region in cfg['regions']:
        folder_path = IG.output_folder_creator(region)
        os.remove(cfg['datasetSettings']['outputSignalFolder'] + region + '_signals.json')
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            logger.error("%s - %s." % (e.filename, e.strerror))

    logger.info('Ending program')
