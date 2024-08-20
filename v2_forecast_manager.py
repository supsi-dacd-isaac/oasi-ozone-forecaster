import json
import logging
import os
import sys
import argparse
import time
import datetime

import urllib3
from influxdb import DataFrameClient

urllib3.disable_warnings()

from classes.forecaster_v2 import ForecasterV2

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

    # Define the forecast type
    # forecast_type = args.t

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
        influx_client = DataFrameClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                        password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                        database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    forecaster = ForecasterV2(influx_client, cfg, logger)
    forecaster.retrieve_predictors()

    if cfg['forecastPeriod']['case'] == 'current':
        forecaster.perform_forecast(cfg['forecastPeriod']['case'])
    else:
        start_day = cfg['forecastPeriod']['startDate']
        end_day = cfg['forecastPeriod']['endDate']

        curr_day = start_day

        end_dt = datetime.datetime.strptime(end_day, '%Y-%m-%d')
        while True:
            # Perform the prediction
            forecaster.perform_forecast(curr_day)
            time.sleep(1)

            # add a day
            curr_dt = datetime.datetime.strptime(curr_day, '%Y-%m-%d')
            curr_day = datetime.datetime.strftime(curr_dt + datetime.timedelta(days=1), '%Y-%m-%d')

            # Last day-1d checking
            if curr_dt.timestamp() >= end_dt.timestamp():
                break

    logger.info('Ending program')