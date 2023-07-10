import json
import logging
import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import urllib3
import datetime
import time
import pytz
import re
import cdsapi
import xarray as xr

from influxdb import InfluxDBClient

urllib3.disable_warnings()


def retrieve_forecast_data(logger, cfg):
    if cfg['CamsSection']['case'] == 'latest':
        start_date = datetime.date.today() - datetime.timedelta(days=1)
        end_date = start_date
    else:
        start_date = datetime.datetime.strptime(cfg['CamsSection']['startDate'], '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(cfg['CamsSection']['endDate'], '%Y-%m-%d').date()

    req_main_pars = cfg['CamsSection']['requestPars']

    c = cdsapi.Client(quiet=True)
    try:
        days_step = datetime.timedelta(days=int(cfg['CamsSection']['daysStep']))
    except Exception as e:
        logger.error('failed to create timedelta out of days_step: {}'.format(str(e)))
        return -1

    logger.info('Retrieving forecast data from %s to %s' % (start_date, end_date))

    must_break = False
    start = time.time()

    while start_date <= end_date and not must_break:
        if start_date + days_step >= end_date:
            # today's data might not be available yet
            curr_end_date = datetime.date.today() - datetime.timedelta(days=1) if end_date == datetime.date.today() else end_date
            must_break = True
            logger.debug('last iteration: setting curr_end_date to yesterday, must_break to True')
        else:
            curr_end_date = start_date + days_step
            logger.debug('end date for next iteration: {}'.format(curr_end_date))

        # Cycle over the regions
        for region in cfg['regions'].keys():

            # Set area request parameter
            req_main_pars['area'] = cfg['regions'][region]['coord']

            # Cycle over the levels
            for level in cfg['CamsSection']['levels']:
                logger.info('Request main parameter -> region: %s, level: %s, period: [%s:%s]' % (region, level,
                                                                                                  start_date,
                                                                                                  curr_end_date))
                # Set level request parameter
                req_main_pars['level'] = str(level)

                err_count = 0

                while err_count < cfg['CamsSection']['maxRetries']:

                    # Set date request parameter
                    req_main_pars['date'] = '{}/{}'.format(start_date, curr_end_date)

                    try:
                        output_file_name = '%s_lvl%i_%s_%s.nc' % (region, level, start_date, curr_end_date)
                        output_file_path = os.path.join(cfg['regions'][region]['inputPath'], output_file_name)

                        c.retrieve(cfg['CamsSection']['dataset'], req_main_pars, output_file_path)

                        logger.info('data retrieved successfully, starting next iteration')
                        logger.debug('data from {} to {} retrieved in {:.2f}s'.format(start_date, end_date, time.time() - start))
                        break
                    except Exception as e:
                        logger.error('failed to retrieve data in dates range {} - {}: {}'.format(start_date, curr_end_date, str(e)))
                        err_count += 1
                        time.sleep(1)

                if err_count == cfg['CamsSection']['maxRetries']:
                    logger.error('max number of retries reached, shutting down')
                    return -1

        start_date = curr_end_date + datetime.timedelta(days=1)

    logger.info('data retrieved')

    return 0


def merge_nc_to_single_dataframe(nc_files, nc_path, logger):
    dfs = []
    regex = re.compile(r'.*(\d{4}-\d{2}-\d{2}).nc')
    for file in nc_files:
        file_loc = os.path.join(nc_path, file)
        with xr.open_dataset(file_loc) as ds:
            df = ds.to_dataframe().reset_index()
            # get year, month and day from file name
            ymd = regex.search(file).group(1).split('-')
            nc_date = datetime.date(int(ymd[0]), int(ymd[1]), int(ymd[2]))

            logger.debug('date of current file: {}'.format(nc_date))

            # insert date column
            df['date'] = nc_date
            df.set_index('date', inplace=True)

            # swap level <-> longitude columns
            cols = list(df)
            cols[1], cols[2] = cols[2], cols[1]
            df = df.loc[:, cols]

            dfs.append(df)

    return pd.concat(dfs)


def get_single_row_dataframe(df_level, d, h):
    str_time = '%i days %02d:00:00' % (d, h)

    # Calculate the mean and get the useful columns
    avg = df_level[df_level.time == str_time][cfg['regions'][region]['columnsToConsider']].mean()
    tmp_df = avg.to_frame().T

    # Set the index
    idx = d * 24 + h
    tmp_df = tmp_df.set_index(pd.Index([idx]))
    return tmp_df


def handle_datafame_single_level(df_level):
    # Cycle over the time
    avg_df = pd.DataFrame()
    for day in np.arange(0, 4):
        for hour in np.arange(0, 24):
            # Set the filtering time-related string
            tmp_df = get_single_row_dataframe(df_level, day, hour)

            # Append the new row
            avg_df = pd.concat([avg_df, tmp_df])

    return avg_df


def append_points(df, str_date, points, region, lvl):
    dt = datetime.datetime.strptime('%sT00:00:00Z' % str_date, '%Y-%m-%dT00:00:00Z')
    dt = pytz.utc.localize(dt)
    ts = int(dt.timestamp())
    for index_row, row in df.iterrows():
        str_step = 'step%02d' % index_row
        for index_col, val in row.items():
            point = {
                'time': ts,
                'measurement': cfg['influxDB']['measurementInputsForecastsCopernicus'],
                'fields': dict(value=float(val)),
                'tags': dict(signal='%s_lvl%i_copern' % (index_col, lvl), location=region, step=str_step)
            }
            points.append(point)

    return points


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

    # STEP1: Download the raw files in .nc format
    if cfg['CamsSection']['downloadEnabling'] is True:
        retrieve_forecast_data(logger, cfg)

    # STEP2: Read the nc files and create the csv
    for region in cfg['regions'].keys():
        logger.info('Starting processing .nc file for region %s' % region)
        nc_files = glob.glob(os.path.join(cfg['regions'][region]['inputPath'], '*.nc'))

        if len(nc_files) == 0:
            logger.error('No .nc file available in %s' % cfg['regions'][region]['inputPath'])
        else:
            logger.info('Find %s .nc files' % len(nc_files))
            df = merge_nc_to_single_dataframe(nc_files, cfg['regions'][region]['inputPath'], logger)
            df.to_csv('%s%s%s_%s_%s.csv' % (cfg['regions'][region]['inputPath'], os.sep, region,
                                            df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')))

            if cfg['CamsSection']['resultFilesDeleting'] is True:
                for nc_file in nc_files:
                    logger.info('Delete NC file %s' % nc_file)
                    os.unlink(nc_file)

    # STEP3: Read the csv and insert data in the DB
    points = []
    for region in cfg['regions'].keys():
        csv_files = glob.glob('%s%s/*.csv' % (cfg['regions'][region]['inputPath'], os.sep))

        # Cycle over the input CSV files
        for csv_file in sorted(csv_files):
            logger.info('Getting data from file %s' % csv_file)
            str_date = csv_file[-14:-4]
            df = pd.read_csv(csv_file)
            df_levels = {}

            # Cycle over the levels
            for lvl in np.unique(df.level.values):
                df_levels[int(lvl)] = df[df['level'] == lvl]
                tmp_df = handle_datafame_single_level(df_levels[lvl])
                points = append_points(tmp_df, str_date, points, region, lvl)

                if len(points) >= cfg['influxDB']['maxLinesPerInsert']:
                    logger.info('Send %i points to InfluxDB server' % len(points))
                    influx_client.write_points(points, time_precision=cfg['influxDB']['timePrecision'])
                    points = []

        if cfg['CamsSection']['resultFilesDeleting'] is True:
            for csv_file in csv_files:
                logger.info('Delete csv file %s' % csv_file)
                os.unlink(csv_file)

    if len(points) > 0:
        logger.info('Send %i points to InfluxDB server' % len(points))
        influx_client.write_points(points, time_precision=cfg['influxDB']['timePrecision'])
        points = []

    logger.info('Ending program')
