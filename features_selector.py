import json
import logging
import os
import sys
import argparse

import urllib3
from influxdb import InfluxDBClient
from multiprocessing import Process

from classes.artificial_features import ArtificialFeatures
from classes.features_analyzer import FeaturesAnalyzer
from classes.inputs_gatherer import InputsGatherer

urllib3.disable_warnings()


# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def fa_process(ig, k_region, target, target_data, cfg, logger):
    fa = FeaturesAnalyzer(ig, forecast_type, cfg, logger)
    fa.dataset_reader(k_region, target_column=[target])
    for key, df in fa.dataFrames.items():
        x_data, y_data, features = fa.dataset_splitter(key, df, target)[:3]
        fa.perform_feature_selection(key, x_data, y_data, features, target, target_data)


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

    af = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    ig = InputsGatherer(influx_client, forecast_type, cfg, logger, af)

    procs = []
    for k_region in cfg['regions'].keys():
        for target in cfg['regions'][k_region]['featuresAnalyzer']['targetColumns'].keys():
            tmp_proc = Process(target=fa_process, args=[ig, k_region, target, cfg['regions'][k_region]['featuresAnalyzer']['targetColumns'][target], cfg, logger])
            tmp_proc.start()
            procs.append(tmp_proc)

    for proc in procs:
        proc.join()

    logger.info('Ending program')
