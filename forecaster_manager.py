# --------------------------------------------------------------------------- #
# Importing section
# --------------------------------------------------------------------------- #

import os
import sys
import argparse
import logging
import json
import glob
import time

from influxdb import InfluxDBClient
from datetime import timedelta, datetime
from multiprocessing import Queue, Process

from classes.forecaster import Forecaster
from classes.inputs_gatherer import InputsGatherer

queue_results = Queue()

#  --------------------------------------------------------------------------- #
# Functions
# -----------------------------------------------------------------------------#
def predictor_process(inputs_gatherer, input_cfg_file, forecast_type, location, model_name, q, cfg, logger):
    dp, pv, paf, uf = perform_single_forecast(inputs_gatherer, input_cfg_file, forecast_type, location, model_name, cfg, logger)

    # Write on the queue
    q.put(
            {
                'day_to_predict': dp,
                'location': location,
                'forecast_type': forecast_type,
                'predictor': model_name,
                'predicted_value': pv,
                'perc_available_features': paf,
                'unavailable_features': uf
            }
        )

def perform_single_forecast(inputs_gatherer, input_cfg_file, forecast_type, location, model_name, cfg, logger):
    logger.info('Launch prediction -> type: %s, location: %s, name: %s' % (forecast_type, location['code'], model_name))

    forecaster = Forecaster(influxdb_client=influx_client, forecast_type=forecast_type, location=location,
                            model_name=model_name, cfg=cfg, logger=logger)

    # Create the inputs dataframe
    forecaster.build_model_input_dataset(inputs_gatherer, input_cfg_file)

    # Perform the prediction
    forecaster.predict(input_cfg_file.replace('inputs', 'predictor').replace('json', 'pkl'))

    return forecaster.day_to_predict,\
           forecaster.predicted_value, \
           forecaster.perc_available_features, \
           forecaster.unavailable_features

def perform_forecast(day_case, forecast_type):

    # set the day_case (current | %Y-%m-%d)
    cfg['dayToForecast'] = day_case

    logger.info('Perform the prediction for day \"%s\"' % cfg['dayToForecast'] )

    # Create the inputs gatherer instance
    inputs_gatherer = InputsGatherer(influxdb_client=influx_client, forecast_type=forecast_type, cfg=cfg, logger=logger)
    # Calculate the day_case-1d O3 values and insert them in the DB
    inputs_gatherer.calc_yesterday_o3_daily_values()

    # Calculate the inputs required by all the models of the configured locations
    inputs_gatherer.build_global_input_dataset()

    # Cycle over the locations
    for location in cfg['locations']:

        # Cycle over the models files
        tmp_folder = '%s%s*%s' % (cfg['folders']['models'], os.sep, forecast_type)

        # Processes creation
        procs = []
        results = []
        logger.info('Predictors will work in %s mode' % cfg['predictionSettings']['operationMode'])
        for input_cfg_file in glob.glob('%s%s/inputs_*.json' % (tmp_folder, os.sep)):

            # Check if the current folder refers to a location configured for the prediction
            if location['code'] in input_cfg_file.split(os.sep)[-2]:
                model_name = input_cfg_file.split('inputs_')[-1].split('.json')[0]

                if cfg['predictionSettings']['operationMode'] == 'parallel':
                    tmp_proc = Process(target=predictor_process, args=[inputs_gatherer, input_cfg_file, forecast_type,
                                                                       location, model_name, queue_results, cfg, logger])
                    tmp_proc.start()
                    procs.append(tmp_proc)
                else:
                    logger.info('Predictors will work in sequence')
                    dp, pv, paf, uf = perform_single_forecast(inputs_gatherer, input_cfg_file, forecast_type, location,
                                                          model_name, cfg, logger)
                    results.append({
                                        'day_to_predict': dp,
                                        'location': location,
                                        'forecast_type': forecast_type,
                                        'predictor': model_name,
                                        'predictor': model_name,
                                        'predicted_value': pv,
                                        'perc_available_features': paf,
                                        'unavailable_features': uf
                                    })

        # Collect the results if teh predictors have worked in parallel mode
        if cfg['predictionSettings']['operationMode'] == 'parallel':
            results = []
            for proc in procs:
                proc.join()

            # Read from the queue
            i = 0
            while True:
                item = queue_results.get()
                results.append(item)
                i += 1
                if i == len(procs):
                    break

        logger.info('Print the predictor results')
        for result in results:
            dp_desc = datetime.fromtimestamp(result['day_to_predict']).strftime('%Y-%m-%d')
            logger.info('[%s;%s;%s;%s] -> predicted max(O3) = %.1f, available features = %.0f%%' % (dp_desc,
                                                                                                    result['location']['code'],
                                                                                                    result['forecast_type'],
                                                                                                    result['predictor'],
                                                                                                    result['predicted_value'],
                                                                                                    result['perc_available_features']))

    # todo check this part is still needed, probably yes but calc_kpis() has to be changed strongly
    # if cfg['dayToForecast'] == 'current':
    #     dm.calc_kpis()


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

    # Perform the forecasts for a specific period/current day
    if cfg['forecastPeriod']['case'] == 'current':
        perform_forecast(cfg['forecastPeriod']['case'], forecast_type)
    else:
        start_day = cfg['forecastPeriod']['startDate']
        end_day = cfg['forecastPeriod']['endDate']

        curr_day = start_day

        end_dt = datetime.strptime(end_day, '%Y-%m-%d')
        while True:
            # perform the prediction
            perform_forecast(curr_day, forecast_type)
            time.sleep(1)

            # add a day
            curr_dt = datetime.strptime(curr_day, '%Y-%m-%d')
            curr_day = datetime.strftime(curr_dt + timedelta(days=1), '%Y-%m-%d')

            # Last day-1d checking
            if curr_dt.timestamp() >= end_dt.timestamp():
                break

    logger.info("Ending program")
