import json
import logging
import os
import sys
import argparse

import numba
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import lightgbm
import ngboost
import xgboost
# from skgarden import RandomForestQuantileRegressor

from multiprocessing import Process
import urllib3
from influxdb import InfluxDBClient
from sklearn.model_selection import KFold

from classes.artificial_features import ArtificialFeatures
from classes.features_analyzer import FeaturesAnalyzer
from classes.inputs_gatherer import InputsGatherer
from pyforecaster.trainer import hyperpar_optimizer, retrieve_cv_results, base_storage_fun

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from pyforecaster.metrics import nmae
from pyforecaster.formatter import Formatter

import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def stop_on_no_improvement_callback(study, trial):
    window = 5
    horizon = window
    if len(study.trials) > window:
        horizon += study.best_trial.number
        last_trials = study.trials[-(window + 1):]

        if last_trials[-1].number >= horizon:
            logger.warning(f"No improvement in last {window} trials. Stopping the optimization.")
            study.stop()


def param_space_fun(trial):
    # Parameters:
    # LightGBM: https://lightgbm.readthedocs.io/en/v3.3.2/Parameters.html
    # todo
    #  - gestione dei pesi
    #  - metrica (adesso nmae)
    param_space = {
                    'learning_rate': trial.suggest_float('learning_rate', low=0.005, high=0.2),
                    'n_estimators': trial.suggest_int('n_estimators', low=50, high=500),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', low=0.5, high=1.0),
                    'num_leaves': trial.suggest_int('num_leaves', low=20, high=400)
                  }
    return param_space


def optimize_model(model, df_X, df_y):
    n_trials = 40
    n_folds = 5

    # Random folds selection
    cv_idxs = []
    for i in range(n_folds):
        tr_idx = np.random.randint(0, 2, len(df_X.index), dtype=bool)
        te_idx = ~tr_idx
        cv_idxs.append((tr_idx, te_idx))
    cv = (f for f in cv_idxs)

    # Sequential folds selection
    # formatter = Formatter()
    # fold_generator = formatter.time_kfcv(df_X.index, n_folds)
    # cv = list(fold_generator)

    study, replies = hyperpar_optimizer(df_X, df_y.iloc[:, [0]], model, n_trials=n_trials, metric=nmae,
                                        cv=cv, param_space_fun=param_space_fun,
                                        hpo_type='one_fold', callbacks=[stop_on_no_improvement_callback])
    trials_df = retrieve_cv_results(study)
    assert trials_df['value'].isna().sum() == 0
    return study


def prepare_df_for_optimization(df_all):
    df_all = df_all.dropna(subset=df_all.columns)
    df_all = df_all.set_index('date')
    df_y = pd.DataFrame(df_all, columns=[target])
    df_X = df_all.drop(cfg['regions'][k_region]['targetColumns'], axis=1)
    return df_X, df_y


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

    cfg_file_name = args.c
    cfg = json.loads(open(cfg_file_name).read())

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

    lgb_params = {
        'task': 'train',
        'boosting': 'gbdt',
        'objective': 'regression',
        'num_leaves': 10,
        'learning_rage': 0.05,
        'metric': {'l2', 'l1'},
        'verbose': -1
    }

    af = ArtificialFeatures(None, forecast_type, cfg, logger)
    ig = InputsGatherer(None, forecast_type, cfg, logger, af)

    procs = []
    # Cycle over the regions
    for k_region in cfg['regions'].keys():
        for target in cfg['regions'][k_region]['optimizer']['targetColumns']:
            fa = FeaturesAnalyzer(ig, forecast_type, cfg, logger)
            fa.dataset_reader(k_region, [target])
            df_all = fa.dataFrames[k_region]['dataset'].head(200)
            # df_all = fa.dataFrames[k_region]['dataset']
            df_X, df_y = prepare_df_for_optimization(df_all)

            # OK
            lgb = lightgbm.LGBMRegressor()
            study = optimize_model(lgb, df_X, df_y)
            logger.info('LGB -> Best score = %f; params = %s' % (study.best_value, study.best_params))

            # OK
            # xgb = xgboost.XGBRegressor()
            # study = optimize_model(xgb, df_X, df_y)
            # logger.info('XGB -> Best score = %f; params = %s' % (study.best_value, study.best_params))

            # todo NOK
            # ngb = ngboost.NGBRegressor()
            # study = optimize_model(ngb, df_X, df_y)
            # logger.info('NGB -> Best score = %f; params = %s' % (study.best_value, study.best_params))

            # todo NOK
            # qrf = RandomForestQuantileRegressor()
            # study = optimize_model(qrf, df_X, df_y)
            # logger.info('QRF -> Best score = %f; params = %s' % (study.best_value, study.best_params))

    logger.info('Ending program')
