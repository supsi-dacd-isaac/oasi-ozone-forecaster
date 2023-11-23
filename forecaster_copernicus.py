import copy
import json
import logging
import datetime
import os
import sys
import argparse
import urllib3
import warnings

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient, DataFrameClient

warnings.filterwarnings("ignore")
urllib3.disable_warnings()


def calc_forecast(influx_client_df, input_measurement, forecast_measurement, region_cfg, day, signals, dps):
    logger.info('Calculate forecast for region %s, day %s' % (region_cfg['region'], day.strftime('%Y-%m-%d')))

    # I go back one day because the forecast of day X is currently provided by Copernicus around 20-30 minutes
    # after midnight of day X+1.
    # Thus the latest available prediction for a generic day Y at 7:30 in the morning of Y (MOR case) is the one
    # related to Y-1, downloaded by Copernicus around 7 hours before.
    str_day = (day - datetime.timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')

    str_locations = '('
    for location in region_cfg['sourceLocations']:
        str_locations = '%slocation=\'%s\' OR ' % (str_locations, location)
    str_locations = '%s)' % str_locations[0:-4]

    # Get data from DB
    query = ("SELECT value, step from %s WHERE signal='%s' AND %s AND time='%s' "
             "GROUP BY location") % (input_measurement, region_cfg['sourceSignal'], str_locations, str_day)
    logger.info('Query: %s' % query)
    res = influx_client_df.query(query)

    for k_signal in signals.keys():
        day_to_store = day.replace(hour=12).replace(minute=0).replace(second=0).replace(microsecond=0)
        day_to_store += datetime.timedelta(days=int(k_signal.split('-d')[-1]))

        df_daily_locs = list()
        for location in region_cfg['sourceLocations']:

            df_tmp = res[(input_measurement, (('location', location),))]
            df_tmp['step_int'] = df_tmp['step'].str.replace('step', '').astype(int)
            df_tmp = df_tmp[signals[k_signal]['fromStep']:signals[k_signal]['toStep'] + 1]

            if region_cfg['aggregations']['daily'] == 'max':
                df_daily_locs.append(np.max(df_tmp['value'].values))
            elif region_cfg['aggregations']['daily'] == 'mean':
                df_daily_locs.append(np.mean(df_tmp['value'].values))
            elif region_cfg['aggregations']['daily'] == 'min':
                df_daily_locs.append(np.min(df_tmp['value'].values))

        if region_cfg['aggregations']['locations'] == 'max':
            predicted_value = np.max(df_daily_locs)
        elif region_cfg['aggregations']['locations'] == 'mean':
            predicted_value = np.mean(df_daily_locs)
        elif region_cfg['aggregations']['locations'] == 'min':
            predicted_value = np.min(df_daily_locs)
        else:
            predicted_value = -999

        point = {
            'time': int(day_to_store.timestamp()),
            'measurement': forecast_measurement,
            'fields': dict(PredictedValue=float(predicted_value)),
            'tags': dict(location=region_cfg['region'], signal=k_signal)
        }
        dps.append(point)

    return dps


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

    logger.info('Starting program')

    logger.info('Connection to InfluxDb server on socket [%s:%s]' % (cfg['influxDB']['host'], cfg['influxDB']['port']))
    try:
        influx_client = InfluxDBClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                       password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                       database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])

        influx_client_df = DataFrameClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                           password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                           database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    input_measurement = cfg['influxDB']['measurementInputsForecastsCopernicus']
    forecast_measurement = cfg['influxDB']['measurementOutputCopernicusForecast']
    str_res = ''
    dps = []
    for region_cfg in cfg['regions']:
        if cfg['forecastPeriod']['case'] == 'today':
            dt = datetime.datetime.now()
            dps = calc_forecast(influx_client_df, input_measurement, forecast_measurement, region_cfg, dt,
                                cfg['signals'], dps)
        else:
            start_day = cfg['forecastPeriod']['startDate']
            end_day = cfg['forecastPeriod']['endDate']

            curr_day = start_day

            end_dt = datetime.datetime.strptime(end_day, '%Y-%m-%d')
            while True:
                # add a day
                curr_dt = datetime.datetime.strptime(curr_day, '%Y-%m-%d')
                curr_day = datetime.datetime.strftime(curr_dt + datetime.timedelta(days=1), '%Y-%m-%d')
                dps = calc_forecast(influx_client_df, input_measurement, forecast_measurement, region_cfg,
                                    curr_dt, cfg['signals'], dps)

                # Last day-1d checking
                if curr_dt.timestamp() >= end_dt.timestamp():
                    break

                # Insert the data in the DB if too many
                if len(dps) >= int(cfg['influxDB']['maxLinesPerInsert']):
                    logger.info('Sent %i points to InfluxDB server' % len(dps))
                    influx_client.write_points(dps, time_precision=cfg['influxDB']['timePrecision'])
                    dps = []

    # Insert the remaining data
    if len(dps) > 0:
        logger.info('Sent %i points to InfluxDB server' % len(dps))
        influx_client.write_points(dps, time_precision=cfg['influxDB']['timePrecision'])

    logger.info('Ending program')
