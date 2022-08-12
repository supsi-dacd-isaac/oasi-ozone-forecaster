import json
import logging
import os
import sys
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import urllib3
from influxdb import DataFrameClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

urllib3.disable_warnings()
from classes.comparison_utils import ComparisonUtils as cu
from classes.model_trainer import ModelTrainer as mt

sns.set_style("ticks")


def do_hist(errs, desc, cfg, hist_pars_code):
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


def do_plot(model, qs, desc, plot_folder, th):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('%s - %s' % (desc, model))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot(quantiles_vals, qs[th]['reliability'], marker='o', markerSize=6)
    ax.plot(quantiles_vals, quantiles_vals, marker='o', markerSize=6)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.xlabel('QUANTILES')
    plt.ylabel('ESTIMATED')
    plt.grid()
    plt.savefig('%s/%s_%s_gt%s.png' % (plot_folder, desc[1:-1].replace(':', '_'), model, th), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('%s - %s' % (desc, model))
    ax.set_xlim([0, 1])
    ax.plot(np.arange(0.1, 1, step=0.1), qs[th]['skill'], marker='o', markerSize=6)
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel('QUANTILES')
    plt.ylabel('QUANTILE SCORE')
    plt.grid()
    plt.savefig('%s/%s_%s_qs_gt%s.png' % (plot_folder, desc[1:-1].replace(':', '_'), model, th), dpi=300)
    plt.close()
    plt.show()


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
    # Example:
    # measured_signals = ['YO3', 'YO3']
    # predicted_signals = ['O3-d0', 'O3-d1']
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

    print('CASE,REGION,TARGET,PREDICTOR,START,END,MAE,RMSE,MAE_gt%s,RMSE_gt%s,MAPE_gt%s,QS50_QRF,'
          'QS50_QRF_gt%s' % (cfg['threshold'], cfg['threshold'], cfg['threshold'], cfg['threshold']))
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
                if flag_pers is False:
                    step = int(predicted_signals[i].split('-')[1][1:]) + 1
                    mae_pers = mean_absolute_error(df_measure.values[step:], df_measure.values[0:-step])
                    rmse_pers = np.sqrt(mean_squared_error(df_measure.values[step:], df_measure.values[0:-step]))

                    print('%s,%s,%s,%s,%s,%s,%.1f,%.1f' % (case, region, predicted_signals[i], 'P03-22-PS',
                                                           start_date, end_date, mae_pers, rmse_pers))
                    flag_pers = True

                for predictor in predictors:
                    # Get forecasts
                    res = calc_ngb_prediction('predictions_ngb', region, predictor, case, predicted_signals[i], start_date, end_date)
                    key = ('predictions_ngb', (('location', region), ('predictor', predictor)))
                    if key in res.keys():
                        # print('%s,%s,%s' % (case, region, predictor))
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

                        mae = mean_absolute_error(df_measure['measure'].values, df_predictors[predictor].values)
                        rmse = np.sqrt(mean_squared_error(df_measure['measure'].values, df_predictors[predictor].values))
                        mae_th, rmse_th = mt.calc_mae_rmse_threshold(df_measure['measure'],
                                                                     df_predictors[predictor], cfg['threshold'])

                        mean_std = np.mean(df_mean_std_predictors_ngb[predictor]['StdDev'].values)
                        max_std = np.max(df_mean_std_predictors_ngb[predictor]['StdDev'].values)

                        mape_th = mt.calc_mape_threshold(df_measure['measure'], df_predictors[predictor],
                                                         cfg['threshold'])

                        # qs_ngb = cu.quantile_scores(df_quantiles_predictors_ngb[predictor].values,
                        #                             df_measure['measure'].values, quantiles_vals)
                        qs_qrf = dict()
                        qs_qrf[0] = cu.quantile_scores(df_quantiles_predictors_qrf[predictor].values,
                                                    df_measure['measure'].values, quantiles_vals, 0)
                        qs_qrf[cfg['threshold']] = cu.quantile_scores(df_quantiles_predictors_qrf[predictor].values,
                                                    df_measure['measure'].values, quantiles_vals, cfg['threshold'])


                        print('%s,%s,%s,%s,%s,%s,%.1f,%.1f,%.1f,%.1f,'
                              '%.1f,%.1f,%.1f' % (case, region, predicted_signals[i], predictor, start_date, end_date,
                                                  mae, rmse, mae_th, rmse_th, mape_th, qs_qrf[0]['qs_50'],
                                                  qs_qrf[cfg['threshold']]['qs_50']))
                                                  # qs_ngb['qs_50'], qs_ngb['mae_rel']*1e2)

                        desc = '[%s:%s:%s:%s]' % (region, case, predicted_signals[i], predictor)
                        do_hist(df_measure['measure'].values - df_predictors[predictor].values.ravel(), desc, cfg, 'errHist')

                        # Additional plot
                        if cfg['doPlot'] is True:
                            # do_plot('NGB', qs_ngb, desc, cfg['plotFolder'])
                            do_plot('QRF', qs_qrf, desc, cfg['plotFolder'], 0)
                            do_plot('QRF', qs_qrf, desc, cfg['plotFolder'], cfg['threshold'])

                do_hist(df_measure['measure'].values, region, cfg, 'measHist')