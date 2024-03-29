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
import datetime

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
def sort_results(prediction_results):
    ordered_res = []
    res_data = {}
    for pred_res in prediction_results:
        if pred_res['flag_best'] is True:
            k = '%s_%s' % (pred_res['region'], pred_res['output_signal'])
            ordered_res.append(k)
            res_data[k] = pred_res
    return sorted(ordered_res), res_data


def select_best_predictor_value(result_prediction):
    if result_prediction['best_predictor'] == 'ngb':
        return result_prediction['ngb_prediction']
    elif result_prediction['best_predictor'] == 'xgb':
        return result_prediction['xgb_prediction']
    elif result_prediction['best_predictor'] == 'lgb':
        return result_prediction['lgb_prediction']
    elif result_prediction['best_predictor'] == 'qrf':
        return result_prediction['qrf_prediction']['quantiles']['perc50']


def upload_best_results(prediction_results):
    logger.info('Send best results to remote FTP server')
    # Create first row (header)
    str_results = 'DAY,STATION,CASE,SIGNAL,PRED,PERC_AV_FEAT'
    for k in prediction_results[0]['qrf_prediction']['thresholds'].keys():
        str_results = '%s,PROB%s' % (str_results, k)
    # for k in prediction_results[0]['qrf_prediction']['quantiles'].keys():
    #     str_results = '%s,QUANTILE_%s' % (str_results, k)
    str_results = '%s\n' % str_results

    # Create first row (measure units)
    str_results = '%s,,,,[ug/m^3],[%%]' % str_results
    for _ in prediction_results[0]['qrf_prediction']['thresholds'].keys():
        str_results = '%s,[%%]' % str_results
    # for _ in prediction_results[0]['qrf_prediction']['quantiles'].keys():
    #     str_results = '%s,[ug/m^3]' % str_results
    str_results = '%s\n' % str_results

    # Order the results
    ordered_res, res_data = sort_results(prediction_results)

    # Create data rows
    if len(ordered_res) > 0:
        dt = datetime.fromtimestamp(res_data[ordered_res[0]]['day_to_predict'])
        for k in ordered_res:
            result = res_data[k]
            if result['flag_best'] is True:
                predicted_value = select_best_predictor_value(result)

                str_results = '%s%s,%s,%s,%s,%.1f,%.0f' % (str_results, dt.strftime('%Y-%m-%d'), result['region'],
                                                           forecast_type, result['output_signal'],
                                                           predicted_value, result['perc_available_features'])
                for k in result['qrf_prediction']['thresholds'].keys():
                    str_results = '%s,%.1f' % (str_results, result['qrf_prediction']['thresholds'][k]*100)

                str_results = '%s\n' % str_results

        # Results file creation
        results_file = '%s_%s.csv' % (dt.strftime('%Y-%m-%d'), forecast_type)
        fw = open('%s%s%s' % (cfg['ftp']['localFolders']['tmp'], os.sep, results_file), 'w')
        fw.write(str_results)
        fw.close()

        # Results file uploading
        dm = DataManager(influx_client, cfg, logger)
        dm.open_ftp_connection()
        dm.upload_file(results_file)
        dm.close_ftp_connection()

        # Results file deletion
        os.unlink('%s%s%s' % (cfg['ftp']['localFolders']['tmp'], os.sep, results_file))
    else:
        logger.warning('No best cases configured, data file will not be created and uploaded to remote FTP server')


def notify_summary(prediction_results):
    logger.info('Alert checking')
    str_err = ''
    str_info = ''

    # Order the results
    ordered_res, res_data = sort_results(prediction_results)

    for k in ordered_res:
        result = res_data[k]
        threshold = cfg['regions'][result['region']]['alarms']['thresholds'][forecast_type]
        if result['flag_prediction'] is True:
            predicted_value = select_best_predictor_value(result)
            if result['perc_available_features'] <= threshold:
                str_err = '%s%s_%s_%s: model %s -> predicted max(O3) = %.1f, probabilities: %s, ' \
                          'quantiles %s, available features %.1f%%, alert_threshold %.1f%%, ' \
                          'flag best=%s\n\n' % (str_info,
                                                result['region'],
                                                result['forecast_type'],
                                                result['output_signal'],
                                                result['predictor'],
                                                predicted_value,
                                                result['qrf_prediction']['thresholds'],
                                                result['qrf_prediction']['quantiles'],
                                                result['perc_available_features'],
                                                threshold,
                                                result['flag_best'])
                str_err = '%s\nVariables that were surrogated:' % str_err
                for uf in result['unavailable_features']:
                    str_err = '%s\n%s' % (str_err, uf)
                str_err = '%s\n\n' % str_err
            else:
                # Only d0 results are notified if there are no problems
                if 'd0' in result['output_signal']:
                    str_info = '%s%s_%s_%s, predictor %s: PRED=%.1f ug/m^3; PROB: ' % (str_info, result['region'],
                                                                                       result['forecast_type'],
                                                                                       result['output_signal'],
                                                                                       result['predictor'],
                                                                                       predicted_value)
                    for kt in result['qrf_prediction']['thresholds'].keys():
                        str_info = '%s %s=%.1f%%,' % (str_info, kt, result['qrf_prediction']['thresholds'][kt]*100)
                    str_info = '%s; flag best=%s\n' % (str_info[:-1], result['flag_best'])
        else:
            str_err = '%s%s_%s_%s: model %s -> prediction not performed' % (str_err,
                                                                            result['region'],
                                                                            result['forecast_type'],
                                                                            result['output_signal'],
                                                                            result['predictor'])

            str_err = '%s\nVariables that cannot be surrogated via mean imputation:' % str_err
            for uf in result['unsurrogable_features']:
                str_err = '%s\n%s' % (str_err, uf)
            str_err = '%s\n\n' % str_err

    # Send Slack message
    if cfg['alerts']['slack']['enabled'] is True:
        slack_client = SlackClient(logger, cfg)

        slack_client.send_alert_message('OZONE FORECASTER SUMMARY', '#000000')

        # Print info message
        if len(str_info) > 0:
            slack_client.send_alert_message(str_info, '#00ff00')

        # Print error message
        if len(str_err) > 0:
            slack_client.send_alert_message(str_err, '#ff0000')


def predictor_process(inputs_gatherer, input_cfg_file, forecast_type, region, output_signal, model_name, q, cfg, logger):
    dp, out_sig, ngb_pred, lgb_pred, xgb_pred, paf, uvf, usf, fp, qfr_pred, fb, bp = perform_single_forecast(inputs_gatherer,
                                                                                                             input_cfg_file,
                                                                                                             forecast_type,
                                                                                                             region,
                                                                                                             output_signal,
                                                                                                             model_name,
                                                                                                             cfg,
                                                                                                             logger)

    # Write on the queue
    q.put(
            {
                'day_to_predict': dp,
                'output_signal': out_sig,
                'region': region['code'],
                'forecast_type': forecast_type,
                'predictor': model_name,
                'ngb_prediction': ngb_pred,
                'lgb_prediction': lgb_pred,
                'xgb_prediction': xgb_pred,
                'perc_available_features': paf,
                'qrf_prediction': qfr_pred,
                'unavailable_features': uvf,
                'unsurrogable_features': usf,
                'flag_prediction': fp,
                'flag_best': fb,
                'best_predictor': bp
            }
        )


def perform_single_forecast(inputs_gatherer, input_cfg_file, forecast_type, region_data, output_signal, model_name, cfg, logger):
    str_day_to_predict = datetime.fromtimestamp(inputs_gatherer.day_to_predict).strftime('%Y-%m-%d')
    logger.info('Launch prediction: [type: %s; location: %s; day to predict: %s; signal to predict: %s; '
                'family: %s]' % (forecast_type, region_data['code'], str_day_to_predict, output_signal, model_name))

    forecaster = Forecaster(influxdb_client=influx_client, forecast_type=forecast_type, region=region_data,
                            output_signal=output_signal, model_name=model_name, cfg=cfg, logger=logger)

    # Create the inputs dataframe
    forecaster.build_model_input_dataset(inputs_gatherer, input_cfg_file, output_signal)

    # Perform the prediction
    forecaster.predict(input_cfg_file.replace('inputs', 'predictor').replace('json', 'pkl'), region_data)

    return forecaster.day_to_predict, forecaster.output_signal, forecaster.ngb_output, \
           forecaster.light_gbm_reg_prediction, forecaster.xg_reg_prediction, forecaster.perc_available_features, \
           forecaster.unavailable_features, forecaster.unsurrogable_features, \
           forecaster.do_prediction, forecaster.qrf_output, forecaster.flag_best, forecaster.best_predictor

def perform_forecast(day_case, forecast_type):

    # set the day_case (current | %Y-%m-%d)
    cfg['dayToForecast'] = day_case

    logger.info('Perform the prediction for day \"%s\"' % cfg['dayToForecast'] )

    # Create the artificial features instance
    artificial_features = ArtificialFeatures(influxdb_client=influx_client, forecast_type=forecast_type, cfg=cfg, logger=logger)

    # Create the inputs gatherer instance
    inputs_gatherer = InputsGatherer(influxdb_client=influx_client, forecast_type=forecast_type, cfg=cfg, logger=logger, artificial_features=artificial_features)

    # Calculate the inputs required by all the models of the configured locations
    inputs_gatherer.build_global_input_dataset()

    # Processes creation
    procs = []
    results = []
    logger.info('Predictors will work in %s mode' % cfg['predictionGeneralSettings']['operationMode'])

    # Cycle over the locations
    for region in cfg['regions'].keys():

        # Cycle over the models files
        tmp_folder = '%s%s*%s' % (cfg['folders']['models'], os.sep, forecast_type)

        for input_cfg_file in glob.glob('%s%s/inputs_*.json' % (tmp_folder, os.sep)):

            # Check if the current folder refers to a location configured for the prediction
            if region in input_cfg_file.split(os.sep)[-2]:
                output_signal, model_name = input_cfg_file.split('inputs_')[-1].split('.json')[0].split('_')

                region_info = {'code': region, 'data': cfg['regions'][region]}
                if cfg['predictionGeneralSettings']['operationMode'] == 'parallel':
                    tmp_proc = Process(target=predictor_process, args=[inputs_gatherer, input_cfg_file, forecast_type,
                                                                       region_info, output_signal, model_name,
                                                                       queue_results, cfg, logger])
                    tmp_proc.start()
                    procs.append(tmp_proc)
                else:
                    dp, out_sig, ngb_pred, lgb_pred, xgb_pred, paf, uvf, usf, fp, qfr_pred, fb, bp = perform_single_forecast(inputs_gatherer,
                                                                                                                             input_cfg_file,
                                                                                                                             forecast_type,
                                                                                                                             region_info,
                                                                                                                             output_signal,
                                                                                                                             model_name,
                                                                                                                             cfg,
                                                                                                                             logger)
                    results.append({
                                        'day_to_predict': dp,
                                        'output_signal': out_sig,
                                        'region': region,
                                        'forecast_type': forecast_type,
                                        'predictor': model_name,
                                        'ngb_prediction': ngb_pred,
                                        'lgb_prediction': lgb_pred,
                                        'xgb_prediction': xgb_pred,
                                        'perc_available_features': paf,
                                        'qrf_prediction': qfr_pred,
                                        'unavailable_features': uvf,
                                        'unsurrogable_features': usf,
                                        'flag_prediction': fp,
                                        'flag_best': fb,
                                        'best_predictor': bp
                                    })

    # Collect the results if the predictors have worked in parallel mode
    if cfg['predictionGeneralSettings']['operationMode'] == 'parallel':
        results = []
        for proc in procs:
            proc.join()

        # Read from the queue
        i = 0
        while True:
            item = queue_results.get()
            logger.info('Received msg N° %.2i (%s / %s / %s / %s)' % (i+1, item['region'], item['forecast_type'],
                                                                      item['output_signal'], item['predictor']))
            results.append(item)
            i += 1
            if i == len(procs):
                logger.info('Received all messages (%i) via queue' % len(procs))
                break

    logger.info('Print the predictors results')
    for result in results:
        dp_desc = datetime.fromtimestamp(result['day_to_predict']).strftime('%Y-%m-%d')
        if result['flag_prediction'] is True:
            logger.info('[%s;%s;%s;%s;%s] -> predictions: [ngb = %.1f, lgb = %.1f, '
                        'xgb = %.1f, qrf = %.1f]' % (dp_desc,
                                                     result['region'],
                                                     result['forecast_type'],
                                                     result['output_signal'],
                                                     result['predictor'],
                                                     result['ngb_prediction'],
                                                     result['lgb_prediction'],
                                                     result['xgb_prediction'],
                                                     result['qrf_prediction']['quantiles']['perc50']))
        else:
            logger.info('[%s;%s;%s;%s;%s] -> prediction not performed' % (dp_desc,
                                                                          result['region'],
                                                                          result['forecast_type'],
                                                                          result['output_signal'],
                                                                          result['predictor']))

    # Check if the result summary has to be notified
    if cfg['forecastPeriod']['case'] == 'current':
        notify_summary(prediction_results=results)

    # Check if the result summary has to be notified
    if cfg['ftp']['sendResults'] is True:
        upload_best_results(prediction_results=results)


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