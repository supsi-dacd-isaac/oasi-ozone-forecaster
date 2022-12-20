# --------------------------------------------------------------------------- #
# Importing section
# --------------------------------------------------------------------------- #

import os
import sys
import argparse
import logging
import json

from classes.alerts import SlackClient
from influxdb import InfluxDBClient, DataFrameClient

from classes.data_manager import DataManager

#  --------------------------------------------------------------------------- #
# Functions
# -----------------------------------------------------------------------------#
def slack_msg():
    slack_client = SlackClient(logger, cfg)
    if bool(dm.files_not_correctly_handled):
        str_err = ''
        for k in dm.files_not_correctly_handled:
            str_err = '%sFailed handling of file %s; Exception: %s\n' % (str_err, k, dm.files_not_correctly_handled[k])
        slack_client.send_alert_message('OZONE FORECASTER - RAW FILES ALARM:\n%s' % str_err, '#ff0000')
    else:
        slack_client.send_alert_message('OZONE FORECASTER - RAW FILES PROPERLY HANDLED', '#00ff00')

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
                                       database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])

        influx_df_client = DataFrameClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                           password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                           database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    dm = DataManager(influx_client, cfg, logger, influx_df_client)

    # Download files from the FTP server
    if cfg['ftp']['enabled'] is True:
        logger.info('Download data from FTP server')
        dm.open_ftp_connection()
        dm.download_remote_files()

    # Insert data into InfluxDB
    if cfg['influxDB']['dataImporting'] is True:
        logger.info('Importing in InfluxDB of raw data related to files in %s' % cfg['ftp']['localFolders']['tmp'])
        dm.insert_data()

    if cfg['calculatedInputsSection']['enabled'] is True:
        logger.info('Calculate artificial data and insert it into input measurements')
        dm.calculate_artificial_data()

    # Delete files correctly handled on the FTP server and close the FTP connection
    if cfg['ftp']['enabled'] is True:
        if cfg['ftp']['deleteRemoteFile'] is True:
            logger.info('Delete handled files from FTP server')
            dm.delete_remote_files()

        dm.close_ftp_connection()

    # Slack alert
    if cfg['alerts']['slack']['enabled'] is True:
        slack_msg()

    logger.info("Ending program")
