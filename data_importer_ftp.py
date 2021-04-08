# --------------------------------------------------------------------------- #
# Importing section
# --------------------------------------------------------------------------- #

import ftplib
import os
import time
import sys
import argparse
import pytz
import logging
import json
import copy

from datetime import datetime
from influxdb import InfluxDBClient

import constants


#  --------------------------------------------------------------------------- #
# Functions
# -----------------------------------------------------------------------------#
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_raw_files(ftp_address, ftp_user, ftp_pwd, ftp_root_folders):
    ftp_root_dirs = '%s/' % cfg['ftp']['remoteFolders']

    if os.path.isdir(tmp_folder) is False:
        os.mkdir(tmp_folder)

    try:
        # perform FTP connection and login
        ftp = ftplib.FTP(ftp_address)
        ftp.login(ftp_user, ftp_pwd)

        for ftp_dir in [ftp_root_folders['measures'], ftp_root_folders['forecasts']]:
            logging.info('Getting files via FTP from %s/%s' % (ftp_address, ftp_dir))
            ftp.cwd('/%s' % ftp_dir)

            # cycle over the remote files
            for file_name in ftp.nlst('*'):
                tmp_local_file = os.path.join(tmp_folder + "/", file_name)
                try:
                    logger.info('%s -> %s/%s/%s' % (file_name, os.getcwd(), tmp_folder, file_name))

                    # get the file from the server
                    with open(tmp_local_file, 'wb') as f:
                        def callback(data):
                            f.write(data)
                        ftp.retrbinary("RETR " + file_name, callback)

                    # delete the remote file
                    # ftp.delete(file_name)
                except Exception as e:
                    logging.error('Downloading exception: %s' % str(e))
    except Exception as e:
        logging.error('Connection exception: %s' % str(e))

    # close the FTP connection
    ftp.close()


def global_signals_handling(file_path, dps, tz_local):
    # Open the file
    f = open(file_path, 'rb')

    # cycling over the data
    for raw_row in f:
        # decode raw bytes
        row = raw_row.decode(constants.ENCODING)

        # Check the row considering the configured global signals
        for global_signal in cfg['globalSignals']:

            # Check if the row contains a RHW value
            if global_signal in row:
                # discard final \r and \n characters
                row = row[:-1]

                # get data array
                data = row.replace(' ', '').split('|')

                # timestamp management
                naive_time = datetime.strptime(data[1], '%Y%m%d000000')
                local_dt = tz_local.localize(naive_time)
                # add an hour to delete the DST influence (OASI timestamps are always in UTC+1 format)
                utc_dt = local_dt.astimezone(pytz.utc) + local_dt.dst()

                point = {
                    'time': int(utc_dt.timestamp()),
                    'measurement': cfg['influxDB']['measurementGlobal'],
                    'fields': dict(value=float(data[3])),
                    'tags': dict(signal=global_signal)
                }
                dps.append(copy.deepcopy(point))

    return dps


def location_signals_handling(file_path, file_name, dps, tz_local):
    # Open the file
    f = open(file_path, 'rb')

    # check ARPA datasets
    raw = f.readline()
    # ARPA case
    if 'arpa' in file_name:
        str_locations = raw.decode(constants.ENCODING)
        arpa_locations_keys = str_locations[:-1].split(';')[1:]
        measurement = cfg['influxDB']['measurementARPA']
    # OASI case
    else:
        # get location code
        [oasi_location_key, stuff] = file_name.split('-')
        measurement = cfg['influxDB']['measurementOASI']

    # signals
    raw = f.readline()
    str_signals = raw.decode(constants.ENCODING)
    signals = str_signals[:-1].split(';')[1:]

    # measure units, not used
    f.readline()

    # cycling over the data
    for raw_row in f:
        # decode raw bytes
        row = raw_row.decode(constants.ENCODING)
        row = row[:-1]

        # get raw date time
        dt = row.split(';')[0]

        # get data array
        data = row.split(';')[1:]

        # timestamp management
        naive_time = datetime.strptime(dt, cfg['local']['timeFormatMeasures'])
        local_dt = tz_local.localize(naive_time)
        # add an hour to delete the DST influence (OASI timestamps are always in UTC+1 format)
        utc_dt = local_dt.astimezone(pytz.utc) + local_dt.dst()

        # calculate the UTC timestamp
        utc_ts = int(utc_dt.timestamp())

        for i in range(0, len(data)):

            # Currently, the status are not taken into account
            if is_float(data[i]) is True and signals[i] != 'status':
                if 'arpa' in file_name:
                    location_tag = constants.LOCATIONS[arpa_locations_keys[i]]
                else:
                    location_tag = constants.LOCATIONS[oasi_location_key]

                point = {
                    'time': utc_ts,
                    'measurement': measurement,
                    'fields': dict(value=float(data[i])),
                    'tags': dict(signal=signals[i], location=location_tag, case='MEASURED')
                }
                dps.append(copy.deepcopy(point))

                if len(dps) >= int(cfg['influxDB']['maxLinesPerInsert']):
                    logger.info('Sent %i points to InfluxDB server' % len(dps))
                    influx_client.write_points(dps, time_precision=cfg['influxDB']['timePrecision'])
                    dps = []
                    time.sleep(0.1)

    return dps


def insert_data():
    logging.info('Started data inserting into DB')
    file_names = os.listdir(tmp_folder)
    dps = []

    tz_local = pytz.timezone(cfg['local']['timeZone'])

    for file_name in file_names:
        file_path = '%s/%s' % (tmp_folder, file_name)
        logging.info('Getting data from %s' % file_path)

        # The signal is related to the entire Ticino canton (currently only RHW signal is handled)
        if 'VOPA' in file_name:
            dps = global_signals_handling(file_path, dps, tz_local)

        # The signal is related to a specific location (belonging either to OASI or to ARPA domains)
        else:
            dps = location_signals_handling(file_path, file_name, dps, tz_local)

    # Send remaining points to InfluxDB
    logger.info('Sent %i points to InfluxDB server' % len(dps))
    influx_client.write_points(dps, time_precision=cfg['influxDB']['timePrecision'])


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

    config_file = args.c
    if os.path.isfile(config_file) is False:
        print('\nATTENTION! Unable to open configuration file %s\n' % config_file)
        sys.exit(1)

    cfg = json.loads(open(args.c).read())
    conns_cfg = json.loads(open(cfg['connectionsFile']).read())
    cfg.update(conns_cfg)

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

    # Get raw files from FTP server
    tmp_folder = 'tmp'

    if cfg['ftp']['enabled'] is True:
        get_raw_files(ftp_address=cfg['ftp']['host'], ftp_user=cfg['ftp']['user'], ftp_pwd=cfg['ftp']['password'],
                      ftp_root_folders=cfg['ftp']['remoteFolders'])

    # Insert data into InfluxDB
    insert_data()

    logger.info("Ending program")