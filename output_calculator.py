import json
import logging
import datetime
import os
import sys
import argparse
import pandas as pd
import urllib3
from influxdb import DataFrameClient

from classes.alerts import SlackClient

import warnings
warnings.filterwarnings("ignore")

urllib3.disable_warnings()


def calc_output(measurement, output_cfg, year, start_day, end_day):
    str_locations = '('
    for location in output_cfg['sourceLocations']:
        str_locations = '%slocation=\'%s\' OR ' % (str_locations, location)
    str_locations = '%s)' % str_locations[0:-4]

    # Query handling the hourly aggregation
    query = "select %s(value) from %s where signal='%s' and %s AND " \
            "time>='%s-%sT00:00:00Z' AND time<='%s-%sT23:59:59Z' " \
            "GROUP BY time(1h), location" % (output_cfg['aggregations']['hourly'], measurement,
                                             output_cfg['sourceSignal'],
                                             str_locations, year, start_day, year, end_day)
    logger.info('Query: %s' % query)
    res = influx_client.query(query)

    df_daily_locs = list()
    for location in output_cfg['sourceLocations']:
        df_tmp = res[(measurement, (('location', location),))]
        df_tmp.columns = [location]
        df_tmp['dt'] = df_tmp.index

        # Grouping handling the daily aggregation
        if output_cfg['aggregations']['daily'] == 'max':
            df_tmp_agg = df_tmp.groupby(pd.Grouper(key='dt', axis=0, freq='D')).max()
        elif output_cfg['aggregations']['daily'] == 'mean':
            df_tmp_agg = df_tmp.groupby(pd.Grouper(key='dt', axis=0, freq='D')).mean()
        elif output_cfg['aggregations']['daily'] == 'min':
            df_tmp_agg = df_tmp.groupby(pd.Grouper(key='dt', axis=0, freq='D')).min()
        df_daily_locs.append(df_tmp_agg)

    df_ret = pd.concat(df_daily_locs, axis=1)

    # Final handling related to the location aggregation level
    if output_cfg['aggregations']['locations'] == 'max':
        return df_ret.max(axis=1)
    elif output_cfg['aggregations']['locations'] == 'mean':
        return df_ret.mean(axis=1)
    elif output_cfg['aggregations']['locations'] == 'min':
        return df_ret.min(axis=1)
    return None


if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # Configuration file
    # --------------------------------------------------------------------------- #
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", help="configuration file")
    arg_parser.add_argument("-p", help="period (today | yesterday | custom)")
    arg_parser.add_argument("-l", help="log file (optional, if empty log redirected on stdout)")
    args = arg_parser.parse_args()

    # Load the main parameters
    config_file = args.c
    if os.path.isfile(config_file) is False:
        print('\nATTENTION! Unable to open configuration file %s\n' % config_file)
        sys.exit(1)

    cfg = json.loads(open(args.c).read())
    period = args.p

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
        influx_client = DataFrameClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                        password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                        database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    measurement = cfg['influxDB']['measurementInputsMeasurements']
    str_res = ''
    for output_cfg in cfg['output']:
        tags = {'signal': output_cfg['targetSignal'], 'location': output_cfg['region']}

        if period == 'today':
            dt = datetime.datetime.now()
            ret_dataset = calc_output(measurement, output_cfg, dt.year, '%02d-%02d' % (dt.month, dt.day),
                                      '%02d-%02d' % (dt.month, dt.day))
            influx_client.write_points(pd.DataFrame({'value': ret_dataset}), measurement, tags=tags)
            logger.info('%s[%s][%s] = %.1f' % (tags['signal'], tags['location'], ret_dataset.index[0].date(),
                                               ret_dataset.values[0]))
            str_res = '%s%s: %s[%s] = %.1f\n' % (str_res, tags['location'], tags['signal'], ret_dataset.index[0].date(),
                                                 ret_dataset.values[0])

        elif period == 'yesterday':
            dt = datetime.datetime.now() - datetime.timedelta(days=1)
            ret_dataset = calc_output(measurement, output_cfg, dt.year, '%02d-%02d' % (dt.month, dt.day),
                                      '%02d-%02d' % (dt.month, dt.day))
            influx_client.write_points(pd.DataFrame({'value': ret_dataset}), measurement, tags=tags)
            logger.info('%s[%s][%s] = %.1f' % (tags['signal'], tags['location'], ret_dataset.index[0].date(),
                                               ret_dataset.values[0]))
            str_res = '%s%s: %s[%s] = %.1f\n' % (str_res, tags['location'], tags['signal'], ret_dataset.index[0].date(),
                                                 ret_dataset.values[0])
        else:
            # Configuration variables
            start_day = cfg['period']['customSettings']['startDay']
            end_day = cfg['period']['customSettings']['endDay']

            for year in cfg['period']['customSettings']['years']:
                ret_dataset = calc_output(measurement, output_cfg, year, start_day, end_day)
                influx_client.write_points(pd.DataFrame({'value': ret_dataset}), measurement, tags=tags)

                logger.info('Finished year %s for output %s, region %s' % (year, output_cfg['targetSignal'],
                                                                           output_cfg['region']))

    # Send results to Slack
    if cfg['alerts']['slack']['enabled'] is True:
        slack_client = SlackClient(logger, cfg)
        slack_client.send_alert_message('%s RESULTS:' % period.upper(), '#000000')
        if len(str_res) > 0:
            slack_client.send_alert_message(str_res, '#00ff00')
        else:
            slack_client.send_alert_message('DATA NOT AVAILABLE', '#ff0000')

    logger.info('Ending program')
