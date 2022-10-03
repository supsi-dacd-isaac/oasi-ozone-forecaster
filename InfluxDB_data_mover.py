# --------------------------------------------------------------------------- #
# Importing section
# --------------------------------------------------------------------------- #

import argparse
import json
import logging
import os
import sys
from datetime import timedelta

import urllib3
from influxdb import DataFrameClient
from influxdb import InfluxDBClient

# --------------------------------------------------------------------------- #
# Functions
# -----------------------------------------------------------------------------#

urllib3.disable_warnings()


def move_data_from_one_measurement_to_another(influxDB_client, measurement_from, measurement_to, loc, sig):
    """Copies tags and fields for a signal from one measurement into another"""

    query = "SELECT * INTO %s FROM %s WHERE location = \'%s\' AND signal = \'%s\' GROUP BY *" % (
        measurement_to, measurement_from, loc, sig)
    # influxDB_client.query(query)


def modify_selected_signals_into_another_measurement(df_client, influxDB_client, measurement_from, location_from,
                                                     signal_from, measurement_to, location_to, signal_to, dd=0, hh=0,
                                                     mm=0, start_time='2014-01-01', end_time='2023-01-01'):
    """Shift in time a specific signal in a specific location and measurement and copy the modified signal into another
    specific measurement, location and signal"""

    query = "SELECT * FROM %s WHERE location = \'%s\' AND signal = \'%s\' AND time >= \'%s\' AND time <= \'%s\'" % (
        measurement_from, location_from, signal_from, start_time, end_time)
    res = df_client.query(query)
    df = res[list(res.keys())[0]]

    df.index = df.index + timedelta(days=dd)
    df.index = df.index + timedelta(hours=hh)
    df.index = df.index + timedelta(minutes=mm)

    data_to_write = [{'measurement': measurement_to,
                      'tags': {
                          'signal': signal_to,
                          'location': location_to},
                      'time': i,
                      'fields': {'value': float(d['value'])},
                      }
                     for i, d in df.iterrows()]
    logger.info('Writing %s from %s into %s' % (signal_to, location_from, location_to))
    # influxDB_client.write_points(data_to_write)


def partially_delete_query(influxDB_client, measurement_from, location_from, signal_from, start_time='2014-01-01',
                           end_time='2100-01-01'):
    """Remove data temporal segment from a series"""

    delete_query = "DELETE FROM %s WHERE location = \'%s\' AND signal = \'%s\' AND time > \'%s\' and time < \'%s\'" % (
        measurement_from, location_from, signal_from, start_time, end_time)
    # influxDB_client.query(delete_query)


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
                                       database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])

        dataframe_client = DataFrameClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                           password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                           database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])

    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    # --------------------------------------------------------------------------- #
    # Start data transformation
    # --------------------------------------------------------------------------- #

    measurement_inputs = cfg["influxDB"]["measurementInputsMeasurements"]
    measurement_forecasts = cfg["influxDB"]["measurementInputsForecasts"]
    location_from = cfg["dataMover"]["locationFrom"]
    location_tmp = cfg["dataMover"]["locationTmp"]
    location_to = cfg["dataMover"]["locationTo"]

    signals_measured = cfg["allMeasuredSignals"]
    signals_forecasted = cfg["allForecastedSignals"]

    locations = {}

    # get locations across all measurements
    for measurement in [cfg['influxDB']['measurementMeteoSuisse'], cfg['influxDB']['measurementOASI']]:
        query_location = "SHOW TAG VALUES FROM %s WITH KEY = location" % (measurement)
        res = influx_client.query(query_location)
        locations[measurement] = [l[1] for l in res.raw['series'][0]['values']]

    for measurement in locations.keys():
        for location in locations[measurement]:
            query_signals = "SHOW TAG VALUES FROM %s WITH KEY = signal WHERE location = \'%s\'" % (
                measurement, location)
            res = influx_client.query(query_signals)
            signals = [l[1] for l in res.raw['series'][0]['values']]
            for signal in signals:
                print(measurement, location, signal)
                if signal in signals_measured:
                    move_data_from_one_measurement_to_another(influx_client, measurement, measurement_inputs, location,
                                                              signal)
                elif signal in signals_forecasted:
                    move_data_from_one_measurement_to_another(influx_client, measurement, measurement_forecasts,
                                                              location, signal)
                else:
                    logger.error('ERROR!!', measurement, location, signal)

    for signal in ["CN", "Gl", "P", "Prec", "RH", "T", "Tdew", "WDvect", "WSvect", "WSgust"]:
        modify_selected_signals_into_another_measurement(dataframe_client, influx_client, measurement_inputs,
                                                         location_from, signal, measurement_inputs, location_tmp,
                                                         signal,
                                                         dd=0, hh=-2, mm=5)

        modify_selected_signals_into_another_measurement(dataframe_client, influx_client, measurement_inputs,
                                                         location_tmp, signal, measurement_inputs, location_to, signal,
                                                         dd=0, hh=0, mm=0)

    for signal in ["NO", "NO2", "NOx", "O3"]:
        modify_selected_signals_into_another_measurement(dataframe_client, influx_client, measurement_inputs,
                                                         location_from,
                                                         signal, measurement_inputs, location_tmp, signal, dd=0,
                                                         hh=-2, mm=-15, end_time='2021-01-01')

        partially_delete_query(influx_client, measurement_inputs, location_from, signal, end_time='2021-01-01')

        modify_selected_signals_into_another_measurement(dataframe_client, influx_client, measurement_inputs,
                                                         location_tmp,
                                                         signal, measurement_inputs, location_from, signal, dd=0,
                                                         hh=0, mm=0, end_time='2021-01-01')

    logger.info("Ending program")
