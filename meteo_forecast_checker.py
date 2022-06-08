# --------------------------------------------------------------------------- #
# Importing section
# --------------------------------------------------------------------------- #

import os
import sys
import argparse
import logging
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from influxdb import InfluxDBClient
from datetime import timedelta, datetime
from multiprocessing import Queue, Process

from classes.forecaster import Forecaster
from classes.alerts import SlackClient, EmailClient
from classes.inputs_gatherer import InputsGatherer
from classes.artificial_features import ArtificialFeatures

from classes.data_manager import DataManager

queue_results = Queue()


#  --------------------------------------------------------------------------- #
# Functions
# -----------------------------------------------------------------------------#
def checking_forecast(day_case, forecast_type, fw):

    # set the day_case (current | %Y-%m-%d)
    cfg['dayToForecast'] = day_case

    logger.info('Perform the prediction for day \"%s\"' % cfg['dayToForecast'] )

    # Create the artificial features instance
    artificial_features = ArtificialFeatures(influxdb_client=influx_client, forecast_type=forecast_type, cfg=cfg, logger=logger)

    # Create the inputs gatherer instance
    inputs_gatherer = InputsGatherer(influxdb_client=influx_client, forecast_type=forecast_type, cfg=cfg, logger=logger, artificial_features=artificial_features)

    # Calculate the inputs required by all the models of the configured locations
    inputs_gatherer.build_meteo_forecast_dataset_for_checking(fw)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # Configuration file
    # --------------------------------------------------------------------------- #
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", help="configuration file")
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

    start_day = cfg['forecastPeriod']['startDate']
    end_day = cfg['forecastPeriod']['endDate']

    fw = open('%sresume.csv' % cfg['outputFolder'], 'w')
    fw.write('DATE,SIGNAL,PRED_ERR,INPUT_FORECAST_MAE,INPUT_FORECAST_MEAN_ERR,INPUT_FORECAST_STD_ERR\n')

    fw_all = open('%sresume_all.csv' % cfg['outputFolder'], 'w')
    fw_all.write('DATE,REGIONE,CASE,PREDICTOR,SIGNAL,RANK,PRED_ERR,INPUT_FORECAST_ERR\n')
    fw_all.close()

    end_dt = datetime.strptime(end_day, '%Y-%m-%d')

    # Cycle over the forecats types
    for forecast_type in ['EVE', 'MOR']:
        curr_day = start_day
        while True:
            # perform the prediction
            checking_forecast(curr_day, forecast_type, fw)
            time.sleep(1)

            # add a day
            curr_dt = datetime.strptime(curr_day, '%Y-%m-%d')
            curr_day = datetime.strftime(curr_dt + timedelta(days=1), '%Y-%m-%d')

            # Last day-1d checking
            if curr_dt.timestamp() >= end_dt.timestamp():
                break

    fw.close()

    # do plot
    # df_res = pd.read_csv(cfg['outputFile'])
    # df_glob = df_res[df_res['SIGNAL'] == 'GLOB']
    #
    # # corr, _ = pearsonr(df_glob['ERR'].values, df_glob['MEAN_ERR'])
    # # print('Pearsons correlation: %.3f' % corr)
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.scatter(df_glob['MAE'].values, df_glob['ABS_ERR'].values, marker='o')
    # plt.grid()
    # plt.show()
    # plt.savefig('%s/%s_%s.png' % (plot_folder, desc[1:-1].replace(':', '_'), model), dpi=300)
    # plt.close()

    logger.info("Ending program")