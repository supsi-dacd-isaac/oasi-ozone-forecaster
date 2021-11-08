import json
import logging
import os
import sys
import shutil

import numpy as np
import urllib3
from influxdb import InfluxDBClient

from classes.artificial_features import ArtificialFeatures
from classes.features_analyzer import FeaturesAnalyzer
from classes.inputs_gatherer import InputsGatherer

urllib3.disable_warnings()

cfg = json.loads(open('conf/oasi_dario.json').read())

# Load the connections parameters and update the config dict with the related values
cfg_conns = json.loads(open(cfg['connectionsFile']).read())
cfg.update(cfg_conns)

# Define the forecast type
forecast_type = 'MOR'

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO)

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

AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
IG = InputsGatherer(influx_client, forecast_type, cfg, logger, AF)
FA = FeaturesAnalyzer(IG, forecast_type, cfg, logger)

# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Start testing
# --------------------------------------------------------------------------- #

# Test using custom signals files

AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
IG = InputsGatherer(influx_client, forecast_type, cfg, logger, AF)
FA = FeaturesAnalyzer(IG, forecast_type, cfg, logger)

for dataset in cfg['datasetSettings']['customJSONSignals']:
    assert os.path.isfile(cfg['datasetSettings']['loadSignalsFolder'] + dataset['filename'])

FA.dataset_creator()

for dataset in cfg['datasetSettings']['customJSONSignals']:
    name = dataset['filename'].split('.')[0]
    folder_path = IG.output_folder_creator(name)
    file_path = '%s%s' % (folder_path, 'dataset.csv')
    assert os.path.isfile(file_path)

print(list(FA.dataFrames.keys()))

for key, df in FA.dataFrames.items():
    x_data, y_data = FA.dataset_splitter(key, df)
    features = x_data.columns.values
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    new_features_custom, importance_custom = FA.perform_feature_selection(x_data, y_data, features)
    print(importance_custom)

logger.info('Ending program')
