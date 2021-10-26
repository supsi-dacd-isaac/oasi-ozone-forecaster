# --------------------------------------------------------------------------- #
# Importing section
# --------------------------------------------------------------------------- #

import os
import sys
import argparse
import logging
import json

from influxdb import InfluxDBClient

from classes.inputs_gatherer import InputsGatherer
from classes.artificial_features import ArtificialFeatures

# --------------------------------------------------------------------------- #
# Functions
# -----------------------------------------------------------------------------#


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------------- #
    # Starting program
    # --------------------------------------------------------------------------- #
    logger.info("Starting program")

    # --------------------------------------------------------------------------- #
    # InfluxDB connection
    # --------------------------------------------------------------------------- #
    logger.info('Connection to InfluxDb server on socket [%s:%s]' % (cfg['influxDB']['host'], cfg['influxDB']['port']))
    try:
        influx_client = InfluxDBClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                       password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                       database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    # Create the artificial features instance
    artificial_features = ArtificialFeatures(influxdb_client=influx_client, forecast_type=forecast_type, cfg=cfg,
                                             logger=logger)

    # Create the inputs gatherer instance
    inputs_gatherer = InputsGatherer(influxdb_client=influx_client, forecast_type=forecast_type, cfg=cfg, logger=logger,
                                     artificial_features=artificial_features)

    built_datasets = []
    read_datasets = []

    # Create all the possible signals for a given region
    # inputs_gatherer.generate_all_signals()

    # Build datasets for all the signal files given
    for signal_file in cfg['datasetSettings']['signalsFiles']:
        tmp_df = inputs_gatherer.build_dataset(signals_file=signal_file)
        built_datasets.append(tmp_df)

    # Read datasets from provided csv files
    # for csv_file in cfg['datasetSettings']['csvFiles']:
    #     tmp_df = inputs_gatherer.read_dataset(csv_file=csv_file)
    #     read_datasets.append(tmp_df)

    logger.info("Ending program")
