import json
import logging
import sys
import pytz
import urllib3
import os
import numpy as np
import pandas as pd
import argparse

from influxdb import InfluxDBClient
from datetime import date, datetime, timedelta

dir_path = os.path.dirname(os.path.realpath(__file__))
path_parent = os.path.dirname(dir_path)
sys.path.insert(0, path_parent)

from classes.inputs_gatherer import InputsGatherer
from classes.artificial_features import ArtificialFeatures
from datetime import date, datetime, timedelta

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

    logger.info("Starting program")

    logger.info('Connection to InfluxDb server on socket [%s:%s]' % (cfg['influxDB']['host'], cfg['influxDB']['port']))
    try:
        influx_client = InfluxDBClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                       password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                       database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    cfg["dayToForecast"] = "2021-06-20"
    cfg["VOC"]["useCorrection"] = True

    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)

    IG = InputsGatherer(influxdb_client=influx_client, forecast_type=forecast_type, cfg=cfg, logger=logger,
                        artificial_features=AF)

    cfg["regions"] = {}
    cfg["regions"]["Testing"] = {
        "MeasureStations": ["CHI", "MEN", "BIO", "LUG", "MS-LUG", "LOC"],
        "ForecastStations": ["P_BIO", "TICIA", "OTL"]
    }

    IG.generate_all_signals()

    # Assert we get the expected results and we didn't break anything
    testing_data = json.loads(open(cfg["datasetSettings"]["outputSignalFolder"] + 'Testing_signals.json').read())
    assert len(testing_data["signals"]) == len(set(testing_data["signals"])) == 2917
    assert "LOC__CN__m0" in testing_data["signals"]
    assert "OTL__GLOB__step0" in testing_data["signals"]
    assert "VOC_Totale" in testing_data["signals"]
    assert "KLO-LUG" in testing_data["signals"]
    assert "IsWeekend" in testing_data["signals"]

    # Delete testing data
    os.remove(cfg['datasetSettings']['outputSignalFolder'] + 'Testing_signals.json')

    logger.info("Ending program")
