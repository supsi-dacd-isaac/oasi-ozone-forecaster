import json
import logging
import sys
import pytz
import urllib3
import os
import numpy as np
import pandas as pd

from influxdb import InfluxDBClient
from datetime import date, datetime, timedelta
from classes.inputs_gatherer import InputsGatherer
from classes.artificial_features import ArtificialFeatures
from datetime import date, datetime, timedelta

urllib3.disable_warnings()

cfg = json.loads(open('../conf/oasi.json').read())

# Load the connections parameters and update the config dict with the related values
cfg_conns = json.loads(open(cfg['connectionsFile']).read())
cfg.update(cfg_conns)

# Define the forecast type
forecast_type = 'MOR'

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                    filename='../logs/dario.log')

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

# Add upper folder so the scripts can modify data at the same level of the scripts
os.chdir('../')

# Define local fake region for the only purpose of testing
cfg["regions"]["Testing"] = {
    "MeasureStations": ["CHI", "MEN", "BIO", "LUG", "MS-LUG", "LOC"],
    "ForecastStations": ["P_BIO", "TICIA", "OTL"]
}

IG.generate_all_signals()

# Assert we get the expected results and we didn't broke anything
testing_data = json.loads(open('conf/signals/Testing_all_signals.json').read())
assert len(testing_data["signals"]) == 3349
assert "LOC__CN__m0" in testing_data["signals"]
assert "OTL__GLOB__step0" in testing_data["signals"]

# Delete testing data
os.remove('conf/signals/Testing_all_signals.json')

logger.info("Ending program")
