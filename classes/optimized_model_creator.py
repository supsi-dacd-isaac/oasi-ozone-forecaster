import copy

import numpy as np
import pandas as pd
import pickle
import os
import json
import glob
import scipy
import xgboost as xgb
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE, LogScore
from xgboost_distribution import XGBDistribution
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix

from classes.qrfr import QuantileRandomForestRegressor as qfrfQuantileRandomForestRegressor

import lightgbm
import shap
import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from pyforecaster.trainer import hyperpar_optimizer, retrieve_cv_results
from pyforecaster.metrics import nmae


class OptimizedModelCreator:
    """
    This class will perform hyperparameters optimization + feature selection + model training
    """

    def __init__(self, dataset, target, region, forecast_type, root_folder, cfg, logger):
        """
        Constructor
        :param dataset: Dataset
        :type dataset: pandas dataframe
        :param target: output target
        :type target: str
        :param region: region
        :type region: str
        :param forecast_type: Forecast type (MOR | EVE)
        :type forecast_type: str
        :param root_folder: Data root folder
        :type root_folder: str
        :param cfg: Main configuraion
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        """
        self.dataset = dataset
        self.target = target
        self.region = region
        self.forecast_type = forecast_type
        self.root_folder = root_folder
        self.cfg = cfg
        self.logger = logger
        self.df_X_all, self.df_y = self.prepare_df_for_optimization()
        self.hp_optimized_result = None
        self.df_X_best = None
        self.lgb = None
        self.qrf = None

    def prepare_df_for_optimization(self,):
        df_all = self.dataset.dropna(subset=self.dataset.columns)
        df_all = df_all.set_index('date')
        df_y = pd.DataFrame(df_all, columns=[self.target])
        df_X = df_all.drop(self.cfg['regions'][self.region]['targetColumns'], axis=1)
        return df_X, df_y

    def stop_on_no_improvement_callback(self, study, trial):
        window = self.cfg['hpoBeforeFS']['noImprovementWindow']
        horizon = window
        if len(study.trials) > window:
            horizon += study.best_trial.number
            last_trials = study.trials[-(window + 1):]

            if last_trials[-1].number >= horizon:
                self.logger.warning(f"No improvement in last {window} trials. Stopping the optimization.")
                study.stop()

    def param_space_fun(self, trial):
        # Example:
        # param_space = {
        #     'learning_rate': trial.suggest_float('learning_rate', low=0.005, high=0.2),
        #     'n_estimators': trial.suggest_int('n_estimators', low=50, high=500),
        #     'colsample_bytree': trial.suggest_float('colsample_bytree', low=0.5, high=1.0),
        #     'num_leaves': trial.suggest_int('num_leaves', low=20, high=400)
        # }

        par_space_cfg = self.cfg['hpoBeforeFS']['paramSpace']
        par_space = {}
        for par_conf in par_space_cfg:
            if par_conf['type'] == 'int':
                par_space[par_conf['name']] = trial.suggest_int(par_conf['name'], low=par_conf['low'],
                                                                high=par_conf['high'])
            elif par_conf['type'] == 'float':
                par_space[par_conf['name']] = trial.suggest_float(par_conf['name'], low=par_conf['low'],
                                                                  high=par_conf['high'])
        return par_space


    @staticmethod
    def get_random_cv_idxs(dataset_len, n_folds):
        # Random folds selection
        cv_idxs = []
        for i in range(n_folds):
            tr_idx = np.random.randint(0, 2, len(dataset_len), dtype=bool)
            te_idx = ~tr_idx
            cv_idxs.append((tr_idx, te_idx))
        return cv_idxs

    @staticmethod
    def get_sequential_cv_idxs(dataset_len, n_folds):
        fold_size = int(dataset_len/n_folds)
        mod_fold = int(dataset_len % n_folds)
        cv_idxs = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = i * fold_size + fold_size
            if i == 0:
                tr_idx = np.concatenate((np.zeros(end_idx-start_idx).astype(int),
                                         np.ones(dataset_len-end_idx).astype(int)), axis=0)
            else:
                tr_idx = np.concatenate((np.ones(start_idx).astype(int),
                                         (np.zeros(end_idx-start_idx)).astype(int),
                                         (np.ones(dataset_len-end_idx).astype(int))), axis=0)
            tr_idx = tr_idx.astype(bool)
            te_idx = ~tr_idx
            cv_idxs.append((tr_idx, te_idx))

        if mod_fold > 0:
            tr_idx = np.concatenate((np.ones(end_idx).astype(int), (np.zeros(mod_fold)).astype(int)), axis=0)
            tr_idx = tr_idx.astype(bool)
            te_idx = ~tr_idx
            cv_idxs.append((tr_idx, te_idx))
        return cv_idxs

    def do_hyperparameters_optimization(self, case):

        if case == 'before_fs':
            df_X_dataset = self.df_X_all
            cfg_hpo = self.cfg['hpoBeforeFS']
        else:
            df_X_dataset = self.df_X_best
            cfg_hpo = self.cfg['hpoAfterFS']

        if self.cfg['cv']['shuffle']:
            cv_idxs = self.get_random_cv_idxs(len(df_X_dataset.index), self.cfg['cv']['folds'])
        else:
            cv_idxs = self.get_sequential_cv_idxs(len(df_X_dataset.index), self.cfg['cv']['folds'])
        cv = (f for f in cv_idxs)

        study, replies = hyperpar_optimizer(df_X_dataset, self.df_y.iloc[:, [0]], lightgbm.LGBMRegressor(),
                                            n_trials=cfg_hpo['trials'], metric=nmae, cv=cv, param_space_fun=self.param_space_fun,
                                            hpo_type=self.cfg['cv']['type'],
                                            callbacks=[self.stop_on_no_improvement_callback])
        trials_df = retrieve_cv_results(study)
        assert trials_df['value'].isna().sum() == 0
        self.hp_optimized_result = study

        # Save results
        self.save_best_result('%shpo_%s' % (self.root_folder, case))

    def save_best_result(self, target_folder):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        with open('%s%s%s_best_pars.json' % (target_folder, os.sep, self.target), 'w') as of:
            json.dump(self.hp_optimized_result.best_params, of)

    def initialize_lgb_model(self, pars):
        lgb = lightgbm.LGBMRegressor()
        lgb.set_params(
            learning_rate=pars['learning_rate'],
            n_estimators=pars['n_estimators'],
            colsample_bytree=pars['colsample_bytree'],
            num_leaves=pars['num_leaves']
        )
        return lgb

    def calculate_shapley_values(self, model, X):
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X)
        shapley_values = np.abs(shap_values).mean(axis=0)
        return shapley_values

    def do_feature_selection(self):
        # Initialize an array to store Shapley values for each feature
        shapley_values = np.zeros(self.df_X_all.shape[1])

        # Set the LightGBM parameters
        if self.cfg['hpoBeforeFS']['enabled'] is True:
            fs_lgb = self.initialize_lgb_model(self.hp_optimized_result.best_params)
        else:
            fs_lgb = self.initialize_lgb_model(self.cfg['fs']['defaultParams'])

        X = self.df_X_all.values
        y = self.df_y.values.ravel()

        kf = KFold(n_splits=self.cfg['cv']['folds'], shuffle=self.cfg['cv']['shuffle'])

        # Perform cross-validation
        k = 1
        self.logger.info('CV started')
        for train_idx, test_idx in kf.split(X):
            self.logger.info('Fold %i/%i' % (k, self.cfg['cv']['folds']))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train the LightGBM regressor
            fs_lgb.fit(X_train, y_train)

            # Calculate Shapley values for the current fold
            fold_shapley_values = self.calculate_shapley_values(fs_lgb, X_test)

            # Accumulate Shapley values across folds
            shapley_values += fold_shapley_values

            k += 1
        self.logger.info('CV ended')

        # Calculate average Shapley values across all folds
        shapley_values /= self.cfg['cv']['folds']

        # Sort the Shapley values in descending order
        sorted_idx = np.argsort(shapley_values)[::-1]

        # Save the most important features
        selected_features = []
        for i in sorted_idx[0:self.cfg['fs']['featuresSelected']]:
            selected_features.append(self.df_X_all.columns[i])

        # Set the dataframe containing only the selected features
        self.df_X_best = self.df_X_all
        for col in self.df_X_all.columns:
            # Check if the feature has not been selected
            if col not in selected_features:
                self.df_X_best = self.df_X_best.drop(col, axis=1)
        self.df_X_best = self.df_X_best.reindex(columns=selected_features)

        # Save the FS results
        self.save_fs_results(sorted_idx, shapley_values)

    def save_fs_results(self, sorted_idx, shapley_values):
        target_folder = '%sfs' % self.root_folder
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self.logger.info('Save FS results in folder %s' % target_folder)

        of = open('%s%s%s_features_rank.csv' % (target_folder, os.sep, self.target), 'w')
        of.write('rank,feature,feature_importance\n')
        rank = 1
        for i in sorted_idx:
            of.write('%i,%s,%f\n' % (rank, self.df_X_all.columns[i], shapley_values[i]))
            rank += 1
        of.close()

        with open('%s%s%s_best_features.json' % (target_folder, os.sep, self.target), 'w') as of:
            json.dump({'signals': list(self.df_X_best.columns)}, of)

    def do_models_training(self):
        tr_lgb = self.initialize_lgb_model(self.hp_optimized_result.best_params)

        self.logger.info('LightGBM training started')
        train_data = lightgbm.Dataset(self.df_X_best, label=self.df_y)
        self.lgb = lightgbm.train(self.cfg['fs']['fixedParams'], train_data,
                                  num_boost_round=self.cfg['fs']['numBoostRound'], keep_training_booster=True,
                                  init_model=tr_lgb)
        self.logger.info('LightGBM training ended')

        # todo the part related to QRF must be improved,trying to reuse the optimized parameters of LightGBM model.
        #  Now only n_estimators is used
        self.logger.info('RFQR training started')
        self.qrf = qfrfQuantileRandomForestRegressor(n_estimators=tr_lgb.n_estimators)
        self.qrf.fit(np.array(self.df_X_best.values), np.array(self.df_y.values).ravel())
        self.logger.info('RFQR training ended')

        self.saving_models_training_results(tr_lgb)

    def saving_models_training_results(self, tr_lgb):
        target_folder = '%smt' % self.root_folder
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self.logger.info('Save MT results in folder %s' % target_folder)

        # Inputs/features file
        files_id = '%s-%s_%s' % (self.cfg['mt']['output'], self.target.split('__')[-1], self.cfg['mt']['family'])

        with open('%s%sinputs_%s.json' % (target_folder, os.sep, files_id), 'w') as of:
            json.dump({'signals': list(self.df_X_best.columns)}, of)

        # Metadata file
        metadata = {
            'general': {
                'region': self.region,
                'case': self.forecast_type,
                'target': self.target,
                'family': self.cfg['mt']['family']
            },
            "modelParameters": {
                "lightGBM": tr_lgb.get_params(),
                "quantileRandomForestRegressor": self.qrf.forest.get_params()
            }
        }
        with open('%s%smetadata_%s.json' % (target_folder, os.sep, files_id), 'w') as of:
            json.dump(metadata, of, indent=2)

        # Models file
        pickle.dump([self.lgb, self.qrf], open('%s%spredictor_%s.pkl' % (target_folder, os.sep, files_id), 'wb'))
