import pandas as pd
import pickle
import os
import json
import copy
import lightgbm
import shap
import numpy as np
import xgboost as xgb

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE

from sklearn.model_selection import KFold
from pyforecaster.trainer import hyperpar_optimizer, retrieve_cv_results
from pyforecaster.metrics import err, squerr, rmse, nmae, mape

from classes.qrfr import QuantileRandomForestRegressor as qfrfQuantileRandomForestRegressor


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
        self.active_hpo = None
        self.METRICS = {'nmae': nmae, 'err': err, 'squerr': squerr, 'rmse': rmse, 'mape': mape}

    def prepare_df_for_optimization(self,):
        df_all = self.dataset.dropna(subset=self.dataset.columns)
        df_all = df_all.set_index('date')
        df_y = pd.DataFrame(df_all, columns=[self.target])
        df_X = df_all.drop(self.cfg['regions'][self.region]['targetColumns'], axis=1)
        return df_X, df_y

    def stop_on_no_improvement_callback(self, study, trial):
        if self.active_hpo == 'before_fs':
            window = self.cfg['hpoBeforeFS']['noImprovementWindow']
        else:
            window = self.cfg['hpoAfterFS']['noImprovementWindow']

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

        self.active_hpo = case
        if case == 'before_fs':
            df_X_dataset = self.df_X_all
            cfg_hpo = self.cfg['hpoBeforeFS']
        else:
            df_X_dataset = self.df_X_best
            cfg_hpo = self.cfg['hpoAfterFS']

        if cfg_hpo['cv']['shuffle']:
            cv_idxs = self.get_random_cv_idxs(len(df_X_dataset.index), cfg_hpo['cv']['folds'])
        else:
            cv_idxs = self.get_sequential_cv_idxs(len(df_X_dataset.index), cfg_hpo['cv']['folds'])
        cv = (f for f in cv_idxs)

        study, replies = hyperpar_optimizer(df_X_dataset, self.df_y.iloc[:, [0]], lightgbm.LGBMRegressor(),
                                            n_trials=cfg_hpo['trials'], metric=self.METRICS[cfg_hpo['metric']],
                                            cv=cv, param_space_fun=self.param_space_fun, hpo_type=cfg_hpo['cv']['type'],
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
            json.dump(self.hp_optimized_result.best_params, of, indent=2)

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

        # LightGBM parameters setting
        # Initialize the pars dictionary with the fixed parameters
        pars = copy.deepcopy(self.cfg['fs']['fixedParams'])
        if self.cfg['hpoBeforeFS']['enabled'] is True:
            # Update the pars dictionary with the optimized parameters
            pars.update(self.hp_optimized_result.best_params)
            fs_lgb = self.initialize_lgb_model(pars)
        else:
            # Update the pars dictionary with the default parameters
            pars.update(self.cfg['fs']['defaultParams'])
            fs_lgb = self.initialize_lgb_model(pars)

        X = self.df_X_all.values
        y = self.df_y.values.ravel()

        kf = KFold(n_splits=self.cfg['fs']['cv']['folds'], shuffle=self.cfg['fs']['cv']['shuffle'])

        # Perform cross-validation
        k = 1
        self.logger.info('CV started')
        for train_idx, test_idx in kf.split(X):
            self.logger.info('Fold %i/%i' % (k, self.cfg['fs']['cv']['folds']))
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
        shapley_values /= self.cfg['fs']['cv']['folds']

        # Sort the Shapley values in descending order
        sorted_idx = np.argsort(shapley_values)[::-1]

        # Save the most important features
        selected_features = []
        for i in sorted_idx[0:self.cfg['fs']['featuresSelected']]:
            selected_features.append(self.df_X_all.columns[i])

        # Set the dataframe containing only the selected features
        self.df_X_best = self.df_X_all[selected_features].copy()

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
            json.dump({'signals': list(self.df_X_best.columns)}, of, indent=2)

    def do_models_training(self):
        # Training data preparation
        train_data_Xy_lgb = lightgbm.Dataset(self.df_X_best, label=self.df_y)
        train_data_X = np.array(self.df_X_best.values)
        train_data_y = np.array(self.df_y.values).ravel()

        # LightGBM training
        self.logger.info('LightGBM training started')
        tr_lgb_reg = self.initialize_lgb_model(self.hp_optimized_result.best_params)
        lgb_reg = lightgbm.train(self.cfg['mt']['fixedParams'], train_data_Xy_lgb, num_boost_round=self.cfg['mt']['numBoostRound'],
                                 keep_training_booster=True, init_model=tr_lgb_reg)
        self.logger.info('LightGBM training ended')

        # RFQR training
        # todo the part related to QRF must be improved,trying to reuse the optimized parameters of LightGBM model.
        #  Now only n_estimators is used
        self.logger.info('RFQR training started')
        qrf_reg = qfrfQuantileRandomForestRegressor(n_estimators=tr_lgb_reg.n_estimators)
        qrf_reg.fit(train_data_X, train_data_y)
        self.logger.info('RFQR training ended')

        # NGB + XGB training
        self.logger.info('NGBoost training started')
        ngb_reg = NGBRegressor(n_estimators=tr_lgb_reg.n_estimators, learning_rate=tr_lgb_reg.learning_rate,
                               Dist=Normal, Base=default_tree_learner, natural_gradient=True, verbose=False, Score=MLE,
                               random_state=500)
        ngb_reg.fit(train_data_X, train_data_y)
        self.logger.info('NGBoost training ended')

        self.logger.info('XGBoost training started')
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=tr_lgb_reg.colsample_bytree,
                                   learning_rate=tr_lgb_reg.learning_rate, max_depth=5, alpha=10,
                                   n_estimators=tr_lgb_reg.n_estimators)
        xgb_reg.fit(train_data_X, train_data_y)
        self.logger.info('XGBoost training ended')

        self.saving_models_training_results(tr_lgb_reg, lgb_reg, qrf_reg, ngb_reg, xgb_reg)

    def saving_models_training_results(self, tr_lgb_reg, lgb_reg, qrf_reg, ngb_reg, xgb_reg):
        target_folder = '%smt' % self.root_folder
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self.logger.info('Save MT results in folder %s' % target_folder)

        # Inputs/features file
        _, output_signal, day_case = self.target.split('__')
        files_id = '%s-%s_%s' % (output_signal[1:], day_case, self.cfg['mt']['family'])

        with open('%s%sinputs_%s.json' % (target_folder, os.sep, files_id), 'w') as of:
            json.dump({'signals': list(self.df_X_best.columns)}, of, indent=2)

        # Metadata file
        ngboost_keys = ['natural_gradient', 'n_estimators', 'learning_rate', 'minibatch_frac', 'col_sample', 'verbose']
        ngboost_pars = {key: ngb_reg.get_params()[key] for key in ngboost_keys if key in ngb_reg.get_params()}
        metadata = {
            'general': {
                'region': self.region,
                'case': self.forecast_type,
                'target': self.target,
                'family': self.cfg['mt']['family']
            },
            "modelParameters": {
                "lightGBM": tr_lgb_reg.get_params(),
                "quantileRandomForestRegressor": qrf_reg.forest.get_params(),
                "ngboost": ngboost_pars,
                "xgboost": xgb_reg.get_params()
            }
        }
        with open('%s%smetadata_%s.json' % (target_folder, os.sep, files_id), 'w') as of:
            json.dump(metadata, of, indent=2)

        # Models file (structure needed for the compatibility with forecast_manager script
        pickle.dump([ngb_reg, qrf_reg, qrf_reg, xgb_reg, lgb_reg], open('%s%spredictor_%s.pkl' % (target_folder, os.sep,
                                                                                                  files_id), 'wb'))
