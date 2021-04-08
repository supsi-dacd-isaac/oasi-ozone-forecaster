# --------------------------------------------------------------------------- #
# Importing section
# --------------------------------------------------------------------------- #

import os
import sys
import argparse
import logging
import json

from influxdb import InfluxDBClient
from glob import glob
from datetime import timedelta, datetime

from classes.data_manager import DataManager
from classes.dataset_builder import DatasetBuilder
from classes.threads_manager import ThreadsManager
from classes.results_manager import ResultsManager

#  --------------------------------------------------------------------------- #
# Functions
# -----------------------------------------------------------------------------#

def perform_forecast(day_case):

    # set the day_case (current | %Y-%m-%d)
    cfg['dayToForecast'] = day_case
    dm.cfg['dayToForecast'] = day_case

    # Calculate the day_case-1d O3 values and insert them in the DB
    dm.calc_yesterday_o3_daily_values()

    logger.info('Perform the prediction for day \"%s\"' % day_case)

    # Create the inputs datasets
    for predictors_folder in glob('%s/*' % cfg['matlab']['forecastersFolder']):
        # cycle over the locations to create the input files for the forecasters
        for location in cfg['locations']:

            sigs_file = '%s/json/%s_%s.json' % (predictors_folder, location['code'], forecast_type)

            # check if the input file exists
            if os.path.isfile(sigs_file):

                dsb = DatasetBuilder(influxdb_client=influx_client, cfg=cfg, logger=logger,
                                     forecast_type=forecast_type, predictors_folder=predictors_folder)
                dsb.build(location=location)

                dsb.save_training_data(location=location)

                tmp = predictors_folder.split(os.path.sep)
                dsb.save(output_file='%s/%s_%s_%s.mat' % (cfg['local']['inputMat'], location['code'], forecast_type,
                                                          tmp[-1]))
            else:
                logger.warning('Signal input file %s does not exist, forecast will not be performed' % sigs_file)

    # Launch the threads forecasts
    thm = ThreadsManager(cfg=cfg, logger=logger, forecast_type=forecast_type)
    thm.run()

    # Cycle over the locations to create the input files for the forecasters
    rm = ResultsManager(influxdb_client=influx_client, cfg=cfg, logger=logger, forecast_type=forecast_type,
                        predicted_day=dsb.day_to_predict)
    rm.handle()
    rm.clear()

    if cfg['dayToForecast'] == 'current':
        dm.calc_kpis()


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
                                       database=cfg['influxDB']['database'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    # Create the DataManager instance
    dm = DataManager(influxdb_client=influx_client, cfg=cfg, logger=logger, forecast_type=forecast_type)

    # OASI/ARPA data handling
    dm.get_files(remote_folder=cfg['ftp']['remoteFolders']['measures'],
                 local_folder=cfg['ftp']['localFolders']['measures'])
    dm.save_measures_data(input_folder=cfg['ftp']['localFolders']['measures'])

    # MeteoSuisse data handling
    dm.get_files(remote_folder=cfg['ftp']['remoteFolders']['forecasts'],
                 local_folder=cfg['ftp']['localFolders']['forecasts'])
    dm.save_forecasts_data(input_folder=cfg['ftp']['localFolders']['forecasts'])

    # Perform the forecasts for a specific period/current day
    if cfg['forecastPeriod']['case'] == 'current':
        perform_forecast(cfg['forecastPeriod']['case'])
    else:
        start_day = cfg['forecastPeriod']['startDate']
        end_day = cfg['forecastPeriod']['endDate']

        curr_day = start_day

        end_dt = datetime.strptime(end_day, '%Y-%m-%d')
        while True:
            # perform the prediction
            perform_forecast(curr_day)

            # add a day
            curr_dt = datetime.strptime(curr_day, '%Y-%m-%d')
            curr_day = datetime.strftime(curr_dt + timedelta(days=1), '%Y-%m-%d')

            # Last day-1d checking
            if curr_dt.timestamp() >= end_dt.timestamp():
                break

    logger.info("Ending program")