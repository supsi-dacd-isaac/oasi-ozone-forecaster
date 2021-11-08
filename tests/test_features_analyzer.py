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

# Add upper folder so the scripts can modify data at the same level of the scripts
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

    cfg['datasetSettings']['startDay'] = '07-10'
    cfg['datasetSettings']['endDay'] = '07-20'
    cfg['datasetSettings']['years'] = [2019, 2021]

    # Test using regions

    cfg['featuresAnalyzer']['datasetCreator'] = 'regions'
    cfg['measuredSignalsStations']['BIO'] = []
    cfg['measuredSignalsStations']['CHI'] = []
    cfg['forecastedSignalsStations']['P_BIO'] = []
    cfg['forecastedSignalsStations']['TICIA'] = []
    FA.dataset_creator()

    for region in cfg['regions']:
        fn = cfg['datasetSettings']['outputSignalFolder'] + region + '_signals.json'
        assert os.path.isfile(fn)
        for sig in cfg['regions'][region]['targetColumn']:
            assert sig in json.loads(open(fn).read())['signals']

    print(list(FA.dataFrames.keys()))

    for key, df in FA.dataFrames.items():
        x_data, y_data = FA.dataset_splitter(df)
        features = x_data.columns.values
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        new_features, importance = FA.perform_feature_selection(x_data, y_data, features)
        print(new_features)

    for region in cfg['regions']:
        folder_path = IG.output_folder_creator(region)
        os.remove(cfg['datasetSettings']['outputSignalFolder'] + region + '_signals.json')
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    # Test using custom signals files

    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    IG = InputsGatherer(influx_client, forecast_type, cfg, logger, AF)
    FA = FeaturesAnalyzer(IG, forecast_type, cfg, logger)

    cfg['datasetSettings']['startDay'] = '07-10'
    cfg['datasetSettings']['endDay'] = '07-20'
    cfg['datasetSettings']['years'] = [2019, 2021]
    cfg['featuresAnalyzer']['datasetCreator'] = 'customJSON'

    for dataset in cfg['datasetSettings']['customJSONSignals']:
        assert os.path.isfile(cfg['datasetSettings']['loadSignalsFolder'] + dataset['filename'])

    FA.dataset_creator()

    for dataset in cfg['datasetSettings']['customJSONSignals']:
        name = dataset['filename'].split('.')[0]
        folder_path = IG.output_folder_creator(name)
        file_path = '%s%s' % (folder_path, 'dataset.csv')
        assert os.path.isfile(file_path)
        os.system('cp %s %s' % (file_path, 'conf/csv/tests/' + name + '.csv'))

    print(list(FA.dataFrames.keys()))

    for key, df in FA.dataFrames.items():
        x_data, y_data = FA.dataset_splitter(key, df)
        features = x_data.columns.values
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        new_features_custom, importance_custom = FA.perform_feature_selection(x_data, y_data, features)
        print(importance_custom)

    for dataset in cfg['datasetSettings']['customJSONSignals']:
        name = dataset['filename'].split('.')[0]
        folder_path = IG.output_folder_creator(name)
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    # Read CSV files

    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    IG = InputsGatherer(influx_client, forecast_type, cfg, logger, AF)
    FA = FeaturesAnalyzer(IG, forecast_type, cfg, logger)

    cfg['datasetSettings']['csvFiles'] = [
        {'filename': 'dataset_BIO_MOR.csv', 'targetColumn': ['BIO__YO3__d1']},
        {'filename': 'dataset_CHI_MOR.csv', 'targetColumn': ['CHI__YO3__d1']}]
    cfg['featuresAnalyzer']['datasetCreator'] = 'CSVreader'

    FA.dataset_creator()

    print(list(FA.dataFrames.keys()))

    for key, df in FA.dataFrames.items():
        x_data, y_data = FA.dataset_splitter(key, df)
        features = x_data.columns.values
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        new_features_reader, importance_reader = FA.perform_feature_selection(x_data, y_data, features)
        print(importance_reader)

    for dataset in cfg['datasetSettings']['csvFiles']:
        fn = cfg['datasetSettings']['loadCsvFolder'] + dataset['filename']
        name = dataset['filename'].split('.')[0]
        df = pd.read_csv(fn)
        assert len(df) == 22
        os.remove(fn)
        folder_path = IG.output_folder_creator(name)
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    logger.info('Ending program')
