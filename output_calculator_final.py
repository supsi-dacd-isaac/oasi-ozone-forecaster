import json
import logging
import datetime
import os
import sys
import argparse
import urllib3
import warnings
import time
import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

import requests
import http

warnings.filterwarnings("ignore")
urllib3.disable_warnings()


def aggregate_df(df, agg_case, freq):
    if agg_case is not False:
        if agg_case == 'max':
            df = df.groupby(pd.Grouper(axis=0, freq=freq)).max()
        elif agg_case == 'mean':
            df = df.groupby(pd.Grouper(axis=0, freq=freq)).mean()
        elif agg_case == 'min':
            df = df.groupby(pd.Grouper(axis=0, freq=freq)).min()
    return df


def calc_target(data_str, day, region_cfg):
    raw_data = data_str.split('\n')
    vals_str = []
    dts = []
    vals = []
    for elem in raw_data:
        if day.strftime('%d.%m.%Y') in elem:
            if elem.split(';')[2] != '':
                vals_str.append(elem)
                (dt_str, val, status, stuff) = elem.split(';')
                dts.append(datetime.datetime.strptime(dt_str, '%d.%m.%Y %H:%M:%S'))
                vals.append(float(val))
    if len(dts) > 0:
        df_tmp = pd.DataFrame(data={'val': vals}, index=dts)

        # Perform te hourly aggregation
        df_tmp = aggregate_df(df_tmp, region_cfg['aggregations']['hourly'], 'H')

        # Perform te daily aggregation
        df_tmp = aggregate_df(df_tmp, region_cfg['aggregations']['daily'], 'D')

        return df_tmp.values[0][0]
    else:
        return None


def calculate_final_target(dt, region_cfg, measurement, download_settings, dps):
    logger.info('Calculate final target for region %s, day %s' % (region_cfg['region'], str(dt.date())))

    agg_values = []
    for location in region_cfg['locations'].keys():
        logger.info('Retrieve data for location %s (region %s)' % (location, region_cfg['region']))
        url = '%s&from=%s&to=%s&location=%s' % (download_settings['url'], str(dt.date()), str(dt.date()),
                                                region_cfg['locations'][location]['code'])

        res = requests.get(url)
        logger.info('Request status code: %i' % res.status_code)

        if download_settings["sleepAfterRequest"] > 0:
            time.sleep(download_settings["sleepAfterRequest"])

        if res.status_code == http.HTTPStatus.OK:
            agg_val = calc_target(res.text, dt, region_cfg)
            if agg_val is not None:
                agg_values.append(calc_target(res.text, dt, region_cfg))
            else:
                logger.warning('Aggregated value not available for location %s, day %s' % (location, str(dt.date())))
        else:
            logger.warning('Aggregated value not available for location %s, day %s' % (location, str(dt.date())))

    if len(agg_values) > 0:
        if region_cfg['aggregations']['locations'] == 'max':
            final_value = np.max(agg_values)
        elif region_cfg['aggregations']['locations'] == 'mean':
            final_value = np.mean(agg_values)
        elif region_cfg['aggregations']['locations'] == 'min':
            final_value = np.min(agg_values)

        point = {
            'time': int(dt.replace(hour=12).replace(minute=0).replace(second=0).replace(microsecond=0).timestamp()),
            'measurement': measurement,
            'fields': dict(value=float(final_value)),
            'tags': dict(location=region_cfg['region'], signal=region_cfg['targetSignal'])
        }
        dps.append(point)
    else:
        logger.warning('No final target available for region %s, day %s, '
                       'locations: %s' % (region_cfg['region'], str(dt.date()), list(region_cfg['locations'].keys())))
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
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    measurement = cfg['influxDB']['measurementPostProcessedTargets']

    for region_cfg in cfg['regions']:
        dps = []

        if cfg['forecastPeriod']['case'] == 'yesterday':
            dt = datetime.datetime.now() - datetime.timedelta(days=1)
            dps = calculate_final_target(dt, region_cfg, measurement, cfg['downloadSection'], dps)
        else:
            start_day = cfg['forecastPeriod']['startDate']
            end_day = cfg['forecastPeriod']['endDate']

            curr_day = start_day

            end_dt = datetime.datetime.strptime(end_day, '%Y-%m-%d')
            while True:
                # add a day
                curr_dt = datetime.datetime.strptime(curr_day, '%Y-%m-%d')
                curr_day = datetime.datetime.strftime(curr_dt + datetime.timedelta(days=1), '%Y-%m-%d')
                dps = calculate_final_target(curr_dt, region_cfg, measurement, cfg['downloadSection'], dps)

                # Last day-1d checking
                if curr_dt.timestamp() >= end_dt.timestamp():
                    break

        logger.info('Sent %i points to InfluxDB server' % len(dps))
        influx_client.write_points(dps, time_precision=cfg['influxDB']['timePrecision'])

    logger.info('Ending program')
