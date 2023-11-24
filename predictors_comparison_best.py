import json
import logging
import os
import sys
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, patches
import urllib3
from influxdb import DataFrameClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.validation import check_consistent_length, check_array

import warnings

warnings.filterwarnings("ignore")

urllib3.disable_warnings()
from classes.comparison_utils import ComparisonUtils as cu

sns.set_style("ticks")


def print_output_stat(region, start_date, end_date, values):
    for interval in cfg['intervals']:
        masked_values, _ = mask_dataset(values, values, interval['limits'][0], interval['limits'][1])
        logger.info('%s,%s,%s,%s,AVG=%.1f,MED=%.1f,STD=%.1f,N=%1d' % (region, interval['label'], start_date, end_date,
                                                                      np.mean(masked_values), np.median(masked_values),
                                                                      np.std(masked_values), len(masked_values)))

def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput


def calc_single_prediction(meas, region, predictor, case, signal, start_date, end_date, quantile=None):
    if quantile is not None:
        query = "select mean(PredictedValue) as prediction from %s " \
                "where signal='%s' and location='%s' and " \
                "predictor='%s' and case='%s' and quantile='%s' and time>='%sT00:00:00Z' and " \
                "time<='%sT23:59:59Z' " \
                "group by time(1d), location, predictor" % (meas, signal, region, predictor,
                                                            case, quantile, start_date, end_date)
    else:
        query = "select mean(PredictedValue) as prediction from %s " \
                "where signal='%s' and location='%s' and " \
                "predictor='%s' and case='%s' and time>='%sT00:00:00Z' and " \
                "time<='%sT23:59:59Z' " \
                "group by time(1d), location, predictor" % (meas, signal, region, predictor,
                                                            case, start_date, end_date)
    # logger.info(query)
    return influx_client.query(query)


def mask_dataset(meas, pred, low, up):
    # Mask definition
    mask_low = meas >= low
    mask_up = meas < up
    mask = mask_low & mask_up

    return meas[mask], pred[mask]


def calc_kpis(meas, pred_single, low, up):
    kpis = {}

    # Datasets filtering
    meas_single = meas
    meas_single, pred_single = mask_dataset(meas_single, pred_single, low, up)

    kpis['mae'] = mean_absolute_error(pred_single, meas_single)
    kpis['rmse'] = np.sqrt(mean_squared_error(pred_single, meas_single))
    kpis['mbe'] = np.mean(pred_single - meas_single)

    kpis['cmae'] = np.sqrt(np.power(kpis['mae'], 2) - np.power(kpis['mbe'], 2))
    kpis['crmse'] = np.sqrt(np.power(kpis['rmse'], 2) - np.power(kpis['mbe'], 2))

    kpis['stdev_meas'] = np.std(meas_single)
    kpis['stdev_pred'] = np.std(pred_single)

    kpis['nmae'] = kpis['mae'] / kpis['stdev_meas']
    kpis['nrmse'] = kpis['rmse'] / kpis['stdev_meas']
    kpis['nmbe'] = kpis['mbe'] / kpis['stdev_meas']

    kpis['ncmae'] = kpis['cmae'] / kpis['stdev_meas']
    kpis['ncrmse'] = kpis['crmse'] / kpis['stdev_meas']

    kpis['mape'] = mean_absolute_percentage_error(meas_single, pred_single) * 1e2
    return kpis


def persistence(step, df_measure, low, up):
    meas = df_measure.values[step:]
    pred = df_measure.values[0:-step]
    meas_mask, pred_mask = mask_dataset(meas, pred, low, up)

    return {
        'mae': mean_absolute_error(meas_mask, pred_mask),
        'mbe': np.mean(pred_mask - meas_mask),
        'rmse': np.sqrt(mean_squared_error(meas_mask, pred_mask)),
        'mape': mean_absolute_percentage_error(meas_mask, pred_mask)*1e2,
    }

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

    try:
        influx_client = DataFrameClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                        password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                        database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)

    logger.info('Starting program')

    # Set the main variables
    measured_signals = cfg['measuredSignals']
    predicted_signals = cfg['predictedSignals']

    regions = cfg['regions']
    cases = cfg['cases']

    start_date = cfg['period']['startDate']
    end_date = cfg['period']['endDate']

    # Get the available predictors
    if cfg['predictorsFilter'] is None:
        query = 'SHOW TAG VALUES FROM predictions_ngb WITH KEY="predictor"'
        res = influx_client.query(query)
        predictors = []
        for elem in res.raw['series'][0]['values']:
            predictors.append(elem[1])
    else:
        predictors = cfg['predictorsFilter']

    pred_kpis = dict()
    pred_pers_kpis = dict()
    targets = dict()
    for region in regions:

        days_to_drop = []
        for dtd in cfg['daysToDrop'][region]:
            dtd_str = '%s 00:00:00+00:00' % dtd
            days_to_drop.append(pd.to_datetime(dtd_str, format='%Y-%m-%d %H:%M:%S%z'))

        flag_pers = False
        for i in range(0, len(measured_signals)):
            for case in cases:
                # Get measured output
                query = "select mean(value) as measure from inputs_measurements " \
                        "where signal='%s' and location='%s' and " \
                        "time>='%sT00:00:00Z' and time<='%sT23:59:59Z' " \
                        "group by time(1d), location" % (measured_signals[i], region, start_date, end_date)

                res = influx_client.query(query)
                df_measure = res[('inputs_measurements', (('location', region),))]

                # Drop configured days from the analysis
                df_measure = df_measure.drop(days_to_drop)

                df_predictors = {}

                print_output_stat(region, start_date, end_date, df_measure.values.ravel())

                # Persistence
                if cfg['printPersistence'] is True and flag_pers is False:
                    step = int(predicted_signals[i].split('-')[1][1:]) + 1
                    for interval in cfg['intervals']:
                        pred_pers_kpis[(region, interval['label'], predicted_signals[i][-2:])] = persistence(step, df_measure, interval['limits'][0], interval['limits'][1])

                for predictor in predictors:
                    for regressor in ['ngb', 'xgb', 'lgb', 'qrf']:
                        if regressor == 'ngb':
                            meas_pred = 'predictions_ngb'
                        elif regressor == 'xgb':
                            meas_pred = 'predictions_xgb'
                        elif regressor == 'lgb':
                            meas_pred = 'predictions_light_gbm'
                        elif regressor == 'qrf':
                            meas_pred = 'predictions_qrf_quantiles'

                        # Get forecasts
                        if regressor == 'qrf':
                            res = calc_single_prediction(meas_pred, region, predictor, case, predicted_signals[i],
                                                         start_date, end_date, 'perc50')
                        else:
                            res = calc_single_prediction(meas_pred, region, predictor, case, predicted_signals[i],
                                                         start_date, end_date)
                        key = (meas_pred, (('location', region), ('predictor', predictor)))

                        if key in res.keys():
                            df_predictors[(predictor, regressor)] = res[key]

                            # Drop configured days from the analysis
                            df_predictors[(predictor, regressor)] = df_predictors[(predictor, regressor)].drop(days_to_drop)

                            meas = df_measure['measure'].values
                            pred = df_predictors[(predictor, regressor)].values.ravel()

                            single_pred_kpis = dict()
                            for interval in cfg['intervals']:
                                try:
                                    single_pred_kpis[interval['label']] = calc_kpis(meas, pred, interval['limits'][0],
                                                                                    interval['limits'][1])
                                except Exception as e:
                                    logger.warning('Problems with data analysis for case %s, region %s, signal %s, '
                                                   'interval [%i:%i], predictor %s' % (case, region,
                                                                                       predicted_signals[i],
                                                                                       interval['limits'][0],
                                                                                       interval['limits'][1],
                                                                                       predictor))

                            # Save the results
                            pred_kpis[(case, region, predicted_signals[i], predictor, regressor)] = single_pred_kpis

            targets[region] = meas
    
    interval = cfg['intervalToConsider']
    kpi = cfg['kpi']
    logger.info('Start of analysis of best predictors')
    sorted_dfs = {}
    for case in cases:
        for region in regions:
            for i in range(0, len(measured_signals)):
                df_best = pd.DataFrame(columns=['ID', 'MAE', 'RMSE', 'MBE', 'MAPE'])
                for k_best, v in pred_kpis.items():
                    if len(pred_kpis[k_best].keys()) > 0:
                        if k_best[0] == case and k_best[1] == region and k_best[2] == predicted_signals[i]:
                            try:
                                k_qrf = (k_best[0], k_best[1], k_best[2], k_best[3], 'qrf')
                                new_row_best = {'ID': (k_best[3], k_best[4]),
                                                'MAE': np.round(pred_kpis[k_best][interval]['mae'], 1),
                                                'RMSE': np.round(pred_kpis[k_best][interval]['rmse'], 1),
                                                'MBE': np.round(pred_kpis[k_best][interval]['mbe'], 1),
                                                'MAPE': np.round(pred_kpis[k_best][interval]['mape'], 1),
                                                'MAE_QRF': np.round(pred_kpis[k_qrf][interval]['mae'], 1),
                                                'RMSE_QRF': np.round(pred_kpis[k_qrf][interval]['rmse'], 1),
                                                'MBE_QRF': np.round(pred_kpis[k_qrf][interval]['mbe'], 1),
                                                'MAPE_QRF': np.round(pred_kpis[k_qrf][interval]['mape'], 1),
                                                'MAPE_DIFF': np.round(pred_kpis[k_best][interval]['mape'], 1) - np.round(pred_kpis[k_qrf][interval]['mape'], 1)}
                            except Exception as e:
                                logger.warning('EXCEPTION: %s' % str(e))
                                logger.warning('Exit program')
                                sys.exit()
                            df_best = pd.concat([df_best, pd.DataFrame([new_row_best])], ignore_index=True)

                if len(df_best) > 0:
                    # Add best case
                    family = cfg['bestCases'][region]['forecaster']['bestLabels'][case]['%s-d%i' % (cfg['output'], i)]['family']
                    predictor = cfg['bestCases'][region]['forecaster']['bestLabels'][case]['%s-d%i' % (cfg['output'], i)]['predictor']
                    k_in_charge_best = (case, region, '%s-d%i' % (cfg['output'], i), family, predictor)
                    k_in_charge_qrf = (case, region, '%s-d%i' % (cfg['output'], i), family, 'qrf')
                    new_row = {'ID': 'BEST',
                               'MAE': np.round(pred_kpis[k_in_charge_best][interval]['mae'], 1),
                               'RMSE': np.round(pred_kpis[k_in_charge_best][interval]['rmse'], 1),
                               'MBE': np.round(pred_kpis[k_in_charge_best][interval]['mbe'], 1),
                               'MAPE': np.round(pred_kpis[k_in_charge_best][interval]['mape'], 1),
                               'MAE_QRF': np.round(pred_kpis[k_in_charge_qrf][interval]['mae'], 1),
                               'RMSE_QRF': np.round(pred_kpis[k_in_charge_qrf][interval]['rmse'], 1),
                               'MBE_QRF': np.round(pred_kpis[k_in_charge_qrf][interval]['mbe'], 1),
                               'MAPE_QRF': np.round(pred_kpis[k_in_charge_qrf][interval]['mape'], 1),
                               'MAPE_DIFF': np.round(pred_kpis[k_in_charge_best][interval]['mape'], 1) - np.round(pred_kpis[k_in_charge_qrf][interval]['mape'], 1)}
                    df_in_charge = pd.DataFrame([new_row])
                    df_best = pd.concat([df_best, df_in_charge], ignore_index=True)

                    # Add persistence case
                    new_pers = {'ID': ('PERS'),
                                'MAE': np.round(pred_pers_kpis[(region, interval, 'd%i' % i)]['mae'], 1),
                                'RMSE': np.round(pred_pers_kpis[(region, interval, 'd%i' % i)]['rmse'], 1),
                                'MBE': np.round(pred_pers_kpis[(region, interval, 'd%i' % i)]['mbe'], 1),
                                'MAPE': np.round(pred_pers_kpis[(region, interval, 'd%i' % i)]['mape'], 1),
                                'MAE_QRF': 0,
                                'RMSE_QRF': 0,
                                'MBE_QRF': 0,
                                'MAPE_QRF': 0,
                                'MAPE_DIFF': 0}
                    df_pers = pd.DataFrame([new_pers])
                    df_best = pd.concat([df_best, df_pers], ignore_index=True)

                    # Manage the dataframe
                    df_best = df_best.set_index('ID')
                    sorted_df_best = df_best.sort_values(by=kpi)

                    of = '%s%s%s_%s_%s-d%i_%s' % (cfg['outputFolder'], os.sep, region, case, cfg['output'], i, kpi)
                    sorted_df_best.to_csv('%s.csv' % of)

                    sorted_dfs['%s_%s_%s-d%i' % (region, case, cfg['output'], i)] = sorted_df_best

    output_resume_file_name = '%s%sresume_%s.xlsx' % (cfg['outputFolder'], os.sep, kpi)
    logger.info('Write resume on file %s' % output_resume_file_name)
    with pd.ExcelWriter(output_resume_file_name) as writer:
        for k in sorted(list(sorted_dfs.keys())):
            if ('MOR' in k) or ('EVE' in k and 'd0' in k):
                sorted_dfs[k].to_excel(writer, sheet_name=k, index=True)

    logger.info('Ending program')