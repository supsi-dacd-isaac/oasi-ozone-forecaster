import json
import logging
import os
import sys
import argparse

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, patches
import urllib3
from influxdb import DataFrameClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

urllib3.disable_warnings()
from classes.comparison_utils import ComparisonUtils as cu
from classes.model_trainer import ModelTrainer as mt

sns.set_style("ticks")


def do_hist_targets(errs, desc, cfg, hist_pars_code):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(desc)
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
    plt.xlabel(cfg['histParams'][hist_pars_code]['xlabel'])
    plt.ylabel('OCCURENCES')
    plt.grid()

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
        ax.set_title(desc)
        plt.xticks(np.arange(cfg['histParams'][hist_pars_code]['xtics']['start'],
                             cfg['histParams'][hist_pars_code]['xtics']['end'],
                             step=cfg['histParams'][hist_pars_code]['xtics']['step']))
        plt.yticks(np.arange(cfg['histParams'][hist_pars_code]['ytics']['start'],
                             cfg['histParams'][hist_pars_code]['ytics']['end'],
                             step=cfg['histParams'][hist_pars_code]['ytics']['step']))
        # plt.hist(err, cfg['histParams'][hist_pars_code]['bins'], facecolor=cfg['histParams'][hist_pars_code]['color'],
        plt.hist(err, cfg['histParams'][hist_pars_code]['bins'],
                 alpha=cfg['histParams'][hist_pars_code]['alpha'])
        plt.xlabel(cfg['histParams'][hist_pars_code]['xlabel'])
        legend_data.append(th['label'])
        plt.legend(legend_data)
        plt.ylabel('OCCURENCES')
        plt.grid()

    # plt.show()
    plt.savefig('%s%s%s_%s.png' % (cfg['plotFolder'], os.sep, desc.replace(':', '_').replace('[', '').replace(']', ''),
                                   cfg['histParams'][hist_pars_code]['fileNameSuffix']), dpi=300)
    plt.close()


def calc_ngb_prediction(meas, region, predictor, case, signal, start_date, end_date):
    query = "select mean(PredictedValue) as prediction from %s " \
            "where signal='%s' and location='%s' and " \
            "predictor='%s' and case='%s' and time>='%sT00:00:00Z' and " \
            "time<='%sT23:59:59Z' " \
            "group by time(1d), location, predictor" % (meas, signal, region, predictor,
                                                        case, start_date, end_date)
    # logger.info(query)
    return influx_client.query(query)


def calc_median(meas, region, predictor, case, signal, start_date, end_date):
    query = "select mean(PredictedValue) as prediction from %s " \
            "where signal='%s' and location='%s' and predictor='%s' and case='%s' and quantile='perc50' " \
            "and time>='%sT00:00:00Z' and time<='%sT23:59:59Z' " \
            "group by time(1d), location, predictor, quantile" % (meas, signal, region, predictor, case,
                                                                  start_date, end_date)
    # logger.info(query)
    res = influx_client.query(query)
    return res[(meas, (('location', region), ('predictor', predictor), ('quantile', 'perc50')))]


def calc_mean_std(meas, region, predictor, case, signal, start_date, end_date):
    query = "select mean(Mean) as Mean, mean(StdDev) as StdDev from %s " \
            "where signal='%s' and location='%s' and predictor='%s' and case='%s' and " \
            "time>='%sT00:00:00Z' and time<='%sT23:59:59Z' " \
            "group by time(1d), location, predictor" % (meas, signal, region, predictor, case,
                                                        start_date, end_date)
    # logger.info(query)
    res = influx_client.query(query)
    return res[(meas, (('location', region), ('predictor', predictor)))]


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
        ax.plot(quantiles_vals, qs[th]['reliability'], marker='o', markerSize=6)
        ax.plot(quantiles_vals, quantiles_vals, marker='o', markerSize=6)
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.xlabel('QUANTILES')
        plt.ylabel('ESTIMATED')
        plt.grid()
        plt.savefig('%s/%s_%s_gt%s.png' % (cfg['plotFolder'], desc[1:-1].replace(':', '_'), 'QRF', th), dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('%s - %s' % (desc, 'QRF'))
        ax.set_xlim([0, 1])
        ax.plot(np.arange(0.1, 1, step=0.1), qs[th]['skill'], marker='o', markerSize=6)
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.xlabel('QUANTILES')
        plt.ylabel('QUANTILE SCORE')
        plt.grid()
        plt.savefig('%s/%s_%s_qs_gt%s.png' % (cfg['plotFolder'], desc[1:-1].replace(':', '_'), 'QRF', th), dpi=300)
        plt.close()

def mask_dataset(meas, pred, low, up):
    # Mask definition
    mask_low = meas >= low
    mask_up = meas < up
    mask = mask_low & mask_up

    return meas[mask], pred[mask]


def calc_kpis(meas, pred, low, up):
    meas, pred = mask_dataset(meas, pred, low, up)

    kpis = {}

    kpis['mae'] = mean_absolute_error(pred, meas)
    kpis['rmse'] = np.sqrt(mean_squared_error(pred, meas))
    kpis['mbe'] = np.mean(pred - meas)

    kpis['cmae'] = np.sqrt(np.power(kpis['mae'], 2) - np.power(kpis['mbe'], 2))
    kpis['crmse'] = np.sqrt(np.power(kpis['rmse'], 2) - np.power(kpis['mbe'], 2))

    kpis['stdev_meas'] = np.std(meas)
    kpis['stdev_pred'] = np.std(pred)

    kpis['nmae'] = kpis['mbe'] / kpis['stdev_meas']
    kpis['nrmse'] = kpis['rmse'] / kpis['stdev_meas']
    kpis['nmbe'] = kpis['mbe'] / kpis['stdev_meas']

    kpis['ncmae'] = kpis['cmae'] / kpis['stdev_meas']
    kpis['ncrmse'] = kpis['crmse'] / kpis['stdev_meas']

    return kpis


def plot_target_kpis(pred_kpis, cfg):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot()
    circle = patches.Circle((0.0, 0.0), radius=1.0, color='black', linestyle='dashed', linewidth=4, fill=False)
    ax.add_patch(circle)
    # legend_data = ['']
    for pred in pred_kpis.keys():
        str_pred = '_'.join(pred)
        if str_pred in cfg['kpiTargetGraph']['pointsToShow']:
            for kpis_set in pred_kpis[pred].keys():
                if pred_kpis[pred][kpis_set]['stdev_pred'] - pred_kpis[pred][kpis_set]['stdev_meas'] >= 0:
                    plt.scatter(np.array(pred_kpis[pred][kpis_set]['ncrmse']), np.array([pred_kpis[pred][kpis_set]['nmbe']]),
                                s=200, label='%s %s %s %s %s ncrmse' % (pred[0], pred[1], pred[2], pred[3], kpis_set))
                    plt.scatter(np.array(pred_kpis[pred][kpis_set]['ncmae']), np.array([pred_kpis[pred][kpis_set]['nmbe']]),
                                s=200, label='%s %s %s %s %s ncmae' % (pred[0], pred[1], pred[2], pred[3], kpis_set))
                else:
                    plt.scatter(np.array(-pred_kpis[pred][kpis_set]['ncrmse']), np.array([pred_kpis[pred][kpis_set]['nmbe']]),
                                s=200, label='%s %s %s %s %s ncrmse' % (pred[0], pred[1], pred[2], pred[3], kpis_set))
                    plt.scatter(np.array(-pred_kpis[pred][kpis_set]['ncmae']), np.array([pred_kpis[pred][kpis_set]['nmbe']]),
                                s=200, label='%s %s %s %s %s ncmae' % (pred[0], pred[1], pred[2], pred[3], kpis_set))

    ax.axis('equal')
    ax.legend(framealpha=1, frameon=True, prop={'weight': 'bold'})
    ax.set_xlim([-2.5, 3.0])
    ax.set_ylim([-1.5, 1.5])
    plt.xticks(np.arange(-1.5, 1.5, 0.25), fontsize=16)
    plt.yticks(np.arange(-1.5, 1.5, 0.25), fontsize=16)
    plt.xlabel('NMAE | NCRMSE [-]', fontsize=18, fontweight='bold')
    plt.ylabel('MBE [-]', fontsize=18, fontweight='bold')
    plt.grid()
    # plt.show()
    plt.savefig('%s/kpis_target.png' % (cfg['plotFolder']), dpi=300)
    plt.close()
    plt.show()


def print_kpis(start_date, end_date, pred_kpis):
    for pred in pred_kpis.keys():
        for kpis_set in pred_kpis[pred].keys():
            str_data = '%s,%s,%s,%s,%s,%s,%s' % (pred[0], pred[1], pred[2], pred[3], start_date, end_date, kpis_set)

            # MAE, RMSE, MBE, CMAE, CRMSE, NMAE, NRMSE, NMBE, NCMAE, NCRMSE
            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['mae'])
            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['rmse'])
            str_data = '%s,%.1f' % (str_data, pred_kpis[pred][kpis_set]['mbe'])
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

    print('CASE,REGION,TARGET,PREDICTOR,START,END,INTERVAL,MAE,RMSE,MBE,CMAE,CRMSE,NMAE,NRMSE,NMBE,NCMAE,NCRMSE')
    pred_kpis = dict()
    for region in regions:
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

                df_predictors = {}
                df_mean_std_predictors_ngb = {}
                df_quantiles_predictors_ngb = {}
                df_median_predictors_qrf = {}
                df_quantiles_predictors_qrf = {}

                # Persistence
                if cfg['printPersistence'] is True and flag_pers is False:
                    step = int(predicted_signals[i].split('-')[1][1:]) + 1
                    mae_pers = mean_absolute_error(df_measure.values[step:], df_measure.values[0:-step])
                    rmse_pers = np.sqrt(mean_squared_error(df_measure.values[step:], df_measure.values[0:-step]))

                    print('%s,%s,%s,%s,%s,%s,PERS,%.1f,%.1f' % (case, region, predicted_signals[i], 'P03-22-PS',
                                                                start_date, end_date, mae_pers, rmse_pers))
                    flag_pers = True

                for predictor in predictors:
                    # Get forecasts
                    res = calc_ngb_prediction('predictions_ngb', region, predictor, case, predicted_signals[i], start_date, end_date)
                    key = ('predictions_ngb', (('location', region), ('predictor', predictor)))
                    if key in res.keys():
                        print('ANALISYS: %s,%s,%s' % (case, region, predictor))
                        df_predictors[predictor] = res[key]

                        # Get NGB quantile forecasts
                        df_quantiles_predictors_ngb[predictor] = calc_quantiles('predictions_ngb_quantiles', region,
                                                                                predictor, case, predicted_signals[i],
                                                                                start_date, end_date)

                        # Get QRF quantile forecasts
                        df_quantiles_predictors_qrf[predictor] = calc_quantiles('predictions_qrf_quantiles', region,
                                                                                predictor, case, predicted_signals[i],
                                                                                start_date, end_date)

                        df_mean_std_predictors_ngb[predictor] = calc_mean_std('predictions_ngb_norm_dist',
                                                                              region, predictor, case,
                                                                              predicted_signals[i],
                                                                              start_date, end_date)

                        single_pred_kpis = dict()
                        for interval in cfg['kpiTargetGraph']['intervals']:
                            single_pred_kpis[interval['label']] = calc_kpis(df_measure['measure'].values,
                                                                            df_predictors[predictor].values.ravel(),
                                                                            interval['limits'][0],
                                                                            interval['limits'][1])
                        pred_kpis[(case, region, predicted_signals[i], predictor)] = single_pred_kpis

                        qs_qrf = dict()
                        for th in cfg['qrfGraphParams']['thresholds']:
                            qs_qrf[th] = cu.quantile_scores(df_quantiles_predictors_qrf[predictor].values,
                                                            df_measure['measure'].values, quantiles_vals, th)

                        # desc = '[%s:%s:%s:%s]' % (region, case, predicted_signals[i], predictor)
                        # do_hist_errors(df_predictors[predictor].values.ravel(), df_measure['measure'].values, desc, cfg, 'errHist')
                        #
                        # # Additional plot
                        # do_qrf_plot(qs_qrf, desc, cfg)
                        

                # do_hist_targets(df_measure['measure'].values, region, cfg, 'measHist')

    print_kpis(start_date, end_date, pred_kpis)
    plot_target_kpis(pred_kpis, cfg)