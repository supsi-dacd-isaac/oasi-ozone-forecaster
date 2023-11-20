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
        print('%s,%s,%s,%s,AVG=%.1f,MED=%.1f,STD=%.1f,N=%1d' % (region, interval['label'], start_date, end_date,
                                                                np.mean(masked_values), np.median(masked_values),
                                                                np.std(masked_values), len(masked_values)))



def print_confusion_matrix(meas, pred, desc, cfg):
    class_meas = ['none'] * len(meas)
    class_pred = ['none'] * len(pred)

    labels = []
    for interval in cfg['confusionMatrix']:
        labels.append(interval['label'])

    for i in range(0, len(meas)):
        for interval in cfg['confusionMatrix']:
            if interval['limits'][0] < meas[i] <= interval['limits'][1]:
                class_meas[i] = interval['label']

            if interval['limits'][0] < pred[i] <= interval['limits'][1]:
                class_pred[i] = interval['label']

    cm = confusion_matrix(class_meas, class_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='g')
    plt.title('CM %s' % desc)
    plt.savefig('%s%s%s_cm.png' % (cfg['plotFolder'], os.sep, desc.replace(':', '_').replace('[', '').replace(']', '')), dpi=300)
    plt.close()


def mean_absolute_percentage_error(y_true, y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average'):
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


def do_hist_targets(errs, desc, cfg, hist_pars_code):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(desc, fontsize=20)
    ax.set_xlim(cfg['histParams'][hist_pars_code]['xlim'])
    ax.set_ylim(cfg['histParams'][hist_pars_code]['ylim'])
    plt.xticks(np.arange(cfg['histParams'][hist_pars_code]['xtics']['start'],
                         cfg['histParams'][hist_pars_code]['xtics']['end'],
                         step=cfg['histParams'][hist_pars_code]['xtics']['step']))
    plt.yticks(np.arange(cfg['histParams'][hist_pars_code]['ytics']['start'],
                         cfg['histParams'][hist_pars_code]['ytics']['end'],
                         step=cfg['histParams'][hist_pars_code]['ytics']['step']))
    plt.hist(errs, cfg['histParams'][hist_pars_code]['bins'], facecolor=cfg['histParams'][hist_pars_code]['color'],
             alpha=cfg['histParams'][hist_pars_code]['alpha'])
    plt.xlabel(cfg['histParams'][hist_pars_code]['xlabel'], fontsize=18)
    plt.ylabel('OCCURENCES', fontsize=18)
    plt.grid()

    plt.savefig('%s%s%s_%s.png' % (cfg['plotFolder'], os.sep, desc.replace(':', '_').replace('[', '').replace(']', ''),
                                   cfg['histParams'][hist_pars_code]['fileNameSuffix']), dpi=300)
    plt.close()


def do_hist_all_targets(targets_data, desc, cfg, hist_pars_code):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(desc, fontsize=20)
    ax.set_xlim(cfg['histParams'][hist_pars_code]['xlim'])
    ax.set_ylim(cfg['histParams'][hist_pars_code]['ylim'])
    plt.xticks(np.arange(cfg['histParams'][hist_pars_code]['xtics']['start'],
                         cfg['histParams'][hist_pars_code]['xtics']['end'],
                         step=cfg['histParams'][hist_pars_code]['xtics']['step']))
    plt.yticks(np.arange(cfg['histParams'][hist_pars_code]['ytics']['start'],
                         cfg['histParams'][hist_pars_code]['ytics']['end'],
                         step=cfg['histParams'][hist_pars_code]['ytics']['step']))
    legend = []
    for k in targets_data.keys():
        plt.hist(targets_data[k], cfg['histParams'][hist_pars_code]['bins'], alpha=cfg['histParams'][hist_pars_code]['alpha'])
        legend.append(k)

    plt.xlabel(cfg['histParams'][hist_pars_code]['xlabel'], fontsize=18)
    plt.ylabel('OCCURENCES', fontsize=18)
    plt.grid()
    plt.legend(legend)

    plt.savefig('%s%s%s_%s.png' % (cfg['plotFolder'], os.sep, desc.replace(':', '_').replace('[', '').replace(']', ''),
                                   cfg['histParams'][hist_pars_code]['fileNameSuffix']), dpi=300)
    plt.close()


def do_hist_errors(pred_all, meas_all, desc, cfg, hist_pars_code):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(cfg['histParams'][hist_pars_code]['xlim'])
    ax.set_ylim(cfg['histParams'][hist_pars_code]['ylim'])

    legend_data = []
    for th in cfg['histParams'][hist_pars_code]['thresholds']:
        meas, pred = mask_dataset(meas_all, pred_all, th['limits'][0], th['limits'][1])
        err = pred - meas
        ax.set_title(desc, fontsize=20)
        plt.xticks(np.arange(cfg['histParams'][hist_pars_code]['xtics']['start'],
                             cfg['histParams'][hist_pars_code]['xtics']['end'],
                             step=cfg['histParams'][hist_pars_code]['xtics']['step']))
        plt.yticks(np.arange(cfg['histParams'][hist_pars_code]['ytics']['start'],
                             cfg['histParams'][hist_pars_code]['ytics']['end'],
                             step=cfg['histParams'][hist_pars_code]['ytics']['step']))
        # plt.hist(err, cfg['histParams'][hist_pars_code]['bins'], facecolor=cfg['histParams'][hist_pars_code]['color'],
        plt.hist(err, cfg['histParams'][hist_pars_code]['bins'],
                 alpha=cfg['histParams'][hist_pars_code]['alpha'])
        plt.xlabel(cfg['histParams'][hist_pars_code]['xlabel'], fontsize=18)
        legend_data.append(th['label'])
        plt.legend(legend_data, fontsize=14)
        plt.ylabel('OCCURENCES', fontsize=18)
        plt.grid()

    # plt.show()
    plt.savefig('%s%s%s_%s.png' % (cfg['plotFolder'], os.sep, desc.replace(':', '_').replace('[', '').replace(']', ''),
                                   cfg['histParams'][hist_pars_code]['fileNameSuffix']), dpi=300)
    plt.close()


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


def calc_quantiles(meas, region, predictor, case, signal, start_date, end_date):
    query = "select mean(PredictedValue) as prediction " \
            "from %s " \
            "where signal='%s' and location='%s' and " \
            "predictor='%s' and case='%s' and time>='%sT00:00:00Z' and " \
            "time<='%sT23:59:59Z' " \
            "group by time(1d), location, predictor, quantile" % (meas, signal, region,
                                                                  predictor, case, start_date, end_date)
    # logger.info(query)
    res = influx_client.query(query)
    return cu.handle_quantiles(res, meas, region, predictor, quantiles)


def do_qrf_plot(qs, desc, cfg):
    for th in qs.keys():
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_title('%s - %s' % (desc, 'QRF'))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.plot(quantiles_vals, qs[th]['reliability'], marker='o', markersize=6)
        ax.plot(quantiles_vals, quantiles_vals, marker='o', markersize=6)
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.xlabel('QUANTILES')
        plt.ylabel('ESTIMATED')
        plt.grid()
        plt.savefig('%s/%s_%s_%s.png' % (cfg['plotFolder'], desc[1:-1].replace(':', '_'), 'QRF', th), dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('%s - %s' % (desc, 'QRF'))
        ax.set_xlim([0, 1])
        ax.plot(np.arange(0.1, 1, step=0.1), qs[th]['skill'], marker='o', markersize=6)
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.xlabel('QUANTILES')
        plt.ylabel('QUANTILE SCORE')
        plt.grid()
        plt.savefig('%s/%s_%s_qs_%s.png' % (cfg['plotFolder'], desc[1:-1].replace(':', '_') , 'QRF', th), dpi=300)
        plt.close()


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


def print_kpis(start_date, end_date, pred_kpis, pred_pers_kpis):
    print('\nCASE,REGION,TARGET,PREDICTOR,REGRESSOR,START,END,INTERVAL,MAE,RMSE,MBE,MAPE,CMAE,CRMSE,NMAE,NRMSE,NMBE,NCMAE,NCRMSE')
    pers_flag = {}
    for pred in pred_kpis.keys():
        for interval in cfg['kpiTargetGraph']['intervals']:
            if (pred[1], pred[2], interval['label']) not in pers_flag.keys() and pred[2][-2:] == 'd0':
                str_data = '*,%s,%s,%s,*,%s,%s,%s' % (pred[1], pred[2], 'PERS', start_date, end_date, interval['label'])
                str_data = '%s,%.1f' % (str_data, pred_pers_kpis[(pred[1], interval['label'], 'd0')]['mae'])
                str_data = '%s,%.1f' % (str_data, pred_pers_kpis[(pred[1], interval['label'], 'd0')]['rmse'])
                str_data = '%s,%.1f' % (str_data, pred_pers_kpis[(pred[1], interval['label'], 'd0')]['mbe'])
                str_data = '%s,%.1f' % (str_data, pred_pers_kpis[(pred[1], interval['label'], 'd0')]['mape'])
                str_data = '%s,*,*,*,*,*,*,*' % str_data
                pers_flag[(pred[1], pred[2], interval['label'])] = True
                print(str_data)

        for kpis_set in pred_kpis[pred].keys():
            str_data = '%s,%s,%s,%s,%s,%s,%s,%s' % (pred[0], pred[1], pred[2], pred[3], pred[4], start_date, end_date,
                                                    kpis_set)

            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['mae'])
            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['rmse'])
            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['mbe'])
            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['mape'])
            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['cmae'])
            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['crmse'])
            str_data = '%s,%.3f' % (str_data, pred_kpis[pred][kpis_set]['nmae'])
            str_data = '%s,%.3f' % (str_data, pred_kpis[pred][kpis_set]['nrmse'])
            str_data = '%s,%.3f' % (str_data, pred_kpis[pred][kpis_set]['nmbe'])
            str_data = '%s,%.3f' % (str_data, pred_kpis[pred][kpis_set]['ncmae'])
            str_data = '%s,%.3f' % (str_data, pred_kpis[pred][kpis_set]['ncrmse'])

            print(str_data)


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
        # predictors = predictors[0:1]
    else:
        predictors = cfg['predictorsFilter']

    quantiles = ['perc10', 'perc20', 'perc30', 'perc40', 'perc50', 'perc60', 'perc70', 'perc80', 'perc90']
    quantiles_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    pred_kpis = dict()
    pred_pers_kpis = dict()
    targets = dict()
    for region in regions:

        days_to_drop = []
        for dtd in cfg['daysToDrop'][region]:
            days_to_drop.append(pd.Timestamp(dtd))

        flag_pers = False
        for i in range(0, len(measured_signals)):
            for case in cases:
                # Get measured output
                query = "select mean(value) as measure from inputs_measurements " \
                        "where signal='%s' and location='%s' and " \
                        "time>='%sT00:00:00Z' and time<='%sT23:59:59Z' " \
                        "group by time(1d), location" % (measured_signals[i], region, start_date, end_date)
                # logger.info(query)
                res = influx_client.query(query)
                df_measure = res[('inputs_measurements', (('location', region),))]
                df_measure = df_measure.drop(days_to_drop)

                df_predictors = {}
                df_mean_std_predictors_ngb = {}
                df_median_predictors_qrf = {}
                df_quantiles_predictors_qrf = {}

                print_output_stat(region, start_date, end_date, df_measure.values.ravel())

                # Persistence
                if cfg['printPersistence'] is True and flag_pers is False:
                    step = int(predicted_signals[i].split('-')[1][1:]) + 1
                    for interval in cfg['intervals']:
                        pred_pers_kpis[(region, interval['label'], predicted_signals[i][-2:])] = persistence(step, df_measure, interval['limits'][0], interval['limits'][1])
                    # flag_pers = True

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
                            # print('ANALISYS: %s,%s,%s' % (case, region, predictor))
                            df_predictors[(predictor, regressor)] = res[key]

                            meas = df_measure['measure'].values
                            pred = df_predictors[(predictor, regressor)].values.ravel()

                            single_pred_kpis = dict()
                            for interval in cfg['intervals']:
                                try:
                                    single_pred_kpis[interval['label']] = calc_kpis(meas, pred, interval['limits'][0],
                                                                                    interval['limits'][1])
                                except Exception as e:
                                    print('WARNING: Data not available for case %s, region %s, signal %s, '
                                          'interval [%i:%i], predictor %s' % (case, region, predicted_signals[i],
                                                                              interval['limits'][0],
                                                                              interval['limits'][1], predictor))

                            # Save the results
                            pred_kpis[(case, region, predicted_signals[i], predictor, regressor)] = single_pred_kpis

            targets[region] = meas
    
    interval = cfg['intervalToConsider']
    kpi = cfg['kpi']
    print('ANALYSIS OF BEST PREDICTORS-REGRESSORS:')
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
                                print('EXCEPTION: %s' % str(e))
                                sys.exit()
                            df_best = pd.concat([df_best, pd.DataFrame([new_row_best])], ignore_index=True)

                if len(df_best) > 0:
                    # Add best case
                    family = cfg['bestCases'][region]['forecaster']['bestLabels'][case]['%s-d%i' % (cfg['output'], i)]['family']
                    predictor = cfg['bestCases'][region]['forecaster']['bestLabels'][case]['%s-d%i' % (cfg['output'], i)]['predictor']
                    k_in_charge_best = (case, region, '%s-d%i' % (cfg['output'], i), family, predictor)
                    k_in_charge_qrf = (case, region, '%s-d%i' % (cfg['output'], i), family, 'qrf')
                    new_row = {'ID': ('BEST', k_in_charge_best[3], k_in_charge_best[4]),
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

    with pd.ExcelWriter('%s%sresume_%s.xlsx' % (cfg['outputFolder'], os.sep, kpi)) as writer:
        for k in sorted(list(sorted_dfs.keys())):
            if ('MOR' in k) or ('EVE' in k and 'd0' in k):
                sorted_dfs[k].to_excel(writer, sheet_name=k, index=True)


