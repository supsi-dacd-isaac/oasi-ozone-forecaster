import sys

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

    def __init__(self, ig, target, region, forecast_type, cfg, logger, override=False):
        """
        Constructor
        :param ig: Inputs gatherer
        :type ig: InputsGatherer object
        :param dataset: Dataset
        :type dataset: pandas dataframe
        :param target: output target
        :type target: str
        :param region: region
        :type region: str
        :param forecast_type: Forecast type (MOR | EVE)
        :type forecast_type: str
        :param cfg: Main configuraion
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        """
        self.ig = ig
        self.dataset = None
        self.target = target
        self.region = region
        self.forecast_type = forecast_type
        self.cfg = cfg
        self.logger = logger
        self.df_X_all = None
        self.df_y = None
        self.hp_optimized_result = None
        self.df_X_best = None
        self.active_hpo = None
        self.METRICS = {'nmae': nmae, 'err': err, 'squerr': squerr, 'rmse': rmse, 'mape': mape}
        self.root_folder = self.get_data_folder(self.region, self.forecast_type, self.cfg)
        self.result_folder = self.get_result_folder(self.region, self.forecast_type, self.target, self.cfg)
        if os.path.exists(self.result_folder) and override is False:
            self.logger.error('Result folder %s already exists' % self.result_folder)
            self.logger.info('Exit program')
            sys.exit(-3)

    @staticmethod
    def get_data_folder(region, forecast_type, cfg):
        return '%s%s_%s_%s%s_%s%s%s' % (cfg['outputFolder'], region, forecast_type,
                                        cfg['dataset']['startYear'], cfg['dataset']['startDay'],
                                        cfg['dataset']['endYear'], cfg['dataset']['endDay'], os.sep)

    @staticmethod
    def get_result_folder(region, forecast_type, target, cfg):
        return '%s%s_%s_%s%s_%s%s%s%s%s%s%s' % (cfg['outputFolder'], region, forecast_type,
                                                cfg['dataset']['startYear'], cfg['dataset']['startDay'],
                                                cfg['dataset']['endYear'], cfg['dataset']['endDay'], os.sep,
                                                cfg['family'], os.sep, target, os.sep)

    def fill_datasets(self, region, target):
        output_dfs = {}

        file_path_df = '%s%s_dataset.csv' % (self.root_folder, self.root_folder.split(os.sep)[-2])
        if not os.path.isfile(file_path_df):
            self.logger.error('File %s does not exist' % file_path_df)
            sys.exit(-1)
        # Read the data file
        tmp_df = pd.read_csv(file_path_df)

        # Filtering on data -> only observations related to output values higher than the limit will be considered
        mask = tmp_df[target] >= self.cfg['regions'][region]['dataToConsiderMinLimit']
        output_dfs[region] = {'dataset': tmp_df[mask], 'targetColumns': target}

        # Select only configured input signals
        input_signals = self.ig.generate_input_signals_codes(region)
        candidate_signals = list(output_dfs[region]['dataset'].columns)

        # Remove date and output from candidates list
        candidate_signals.remove('date')
        for target_column in self.cfg['regions'][region]['targetColumns']:
            candidate_signals.remove(target_column)

        # Select candidate signals from data
        for candidate_signal in candidate_signals:
            if candidate_signal not in input_signals:
                # This signal has not to be used in the grid search
                output_dfs[region]['dataset'] = output_dfs[region]['dataset'].drop(candidate_signal, axis=1)

        self.dataset = output_dfs[region]['dataset']
        self.df_X_all, self.df_y = self.prepare_Xy_dfs()

    def nans_management(self):
        # Nans management
        df_tmp = copy.deepcopy(self.dataset)
        df_tmp.dropna(inplace=True)

        self.logger.info('Found %i nans on %i observations (%.0f%%)' % (len(self.dataset)-len(df_tmp),
                                                                        len(self.dataset),
                                                                        ((1-len(df_tmp)/len(self.dataset))*1e2)))

        self.logger.info('Apply "%s" management on nans (available: drop|interpolate|ffill|bfill)' %
                         self.cfg['dataset']['nansManagement'])
        df_all = copy.deepcopy(self.dataset)
        if self.cfg['dataset']['nansManagement'] == 'drop':
            df_all.dropna(inplace=True)
        elif self.cfg['dataset']['nansManagement'] == 'interpolate':
            df_all = df_all.interpolate(method='linear')
        elif self.cfg['dataset']['nansManagement'] == 'ffill':
            df_all = df_all.ffill()
        elif self.cfg['dataset']['nansManagement'] == 'bfill':
            df_all = df_all.bfill()
        else:
            self.logger.info('"%s" nans management not available: drop option is chosen' %
                             self.cfg['dataset']['nansManagement'])
            df_all.dropna(inplace=True)

        return df_all

    def prepare_Xy_dfs(self):
        # Manage the (eventual) nans in the dataset
        df_all = self.nans_management()

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

        # todo weights management is not applied during the hyperparameters optimization because the currently
        #  used version of pyforecaster (0.1) does not support it with 'full' hpo_type
        # Manage the samples weights
        # sample_weights = self.create_weights(self.df_y.iloc[:, [0]][self.target].values)

        study, replies = hyperpar_optimizer(df_X_dataset, self.df_y.iloc[:, [0]], lightgbm.LGBMRegressor(),
                                            n_trials=cfg_hpo['trials'], metric=self.METRICS[cfg_hpo['metric']],
                                            cv=cv, param_space_fun=self.param_space_fun, hpo_type=cfg_hpo['cv']['type'],
                                            callbacks=[self.stop_on_no_improvement_callback])
        trials_df = retrieve_cv_results(study)
        assert trials_df['value'].isna().sum() == 0
        self.hp_optimized_result = study

        # Save results
        self.save_best_result(case, cfg_hpo['metric'])

    def save_best_result(self, case, metric):
        target_folder = '%shpo_%s' % (self.result_folder, case)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        best_trail_data = {
            'metric': metric,
            'value': self.hp_optimized_result.best_trial.value,
            'values': self.hp_optimized_result.best_trial.values,
            'params': self.hp_optimized_result.best_trial.params,
        }
        with open('%s%s%s_%s_best_trial.json' % (target_folder, os.sep, self.cfg['family'], self.target), 'w') as of:
            json.dump(best_trail_data, of, indent=2)

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

    def create_weights(self, y):
        weights = np.ones(len(y))
        for k, v in self.cfg['regions'][self.region]['weights'][self.forecast_type].items():
            if k[0] == '>':
                weights[y > float(k[1:])] = float(v)
            elif k[0] == '<':
                weights[y < float(k[1:])] = float(v)
            elif k[0] == '=':
                weights[y == float(k[1:])] = float(v)
            else:
                self.logger.error('Unable to apply weights for configuration: [%s,%.1f]' % (k, v))
                return None
        return weights

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

            # Manage the samples weights
            sample_weights = self.create_weights(y_train)

            # Train the LightGBM regressor
            fs_lgb.fit(X_train, y_train, sample_weight=sample_weights)

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
        target_folder = '%sfs' % self.result_folder
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self.logger.info('Save FS results in folder %s' % target_folder)

        of = open('%s%s%s_%s_features_rank.csv' % (target_folder, os.sep, self.cfg['family'], self.target), 'w')
        of.write('rank,feature,feature_importance\n')
        rank = 1
        for i in sorted_idx:
            of.write('%i,%s,%f\n' % (rank, self.df_X_all.columns[i], shapley_values[i]))
            rank += 1
        of.close()

        with open('%s%s%s_%s_best_features.json' % (target_folder, os.sep, self.cfg['family'], self.target), 'w') as of:
            json.dump({'signals': list(self.df_X_best.columns)}, of, indent=2)

    def do_models_training(self):
        # Training data preparation
        train_data_X = np.array(self.df_X_best.values)
        train_data_y = np.array(self.df_y.values).ravel()
        sample_weights = self.create_weights(train_data_y)

        train_data_Xy_lgb = lightgbm.Dataset(self.df_X_best, label=self.df_y, weight=sample_weights)

        # LightGBM training
        self.logger.info('LightGBM training started')
        lgb_pars = copy.deepcopy(self.cfg['mt']['fixedParams'])
        lgb_pars.update(self.hp_optimized_result.best_params)
        lgb_reg = lightgbm.train(lgb_pars, train_data_Xy_lgb)
        self.logger.info('LightGBM training ended')

        # Only mae and rmse are now properly translated for XGB and QRF regressors
        if lgb_reg.params['metric'] == 'mae':
            qrf_criterion = 'absolute_error'
            # Deprecated
            xgb_obj = 'reg:linear'
        elif lgb_reg.params['metric'] == 'rmse':
            qrf_criterion = 'squared_error'
            xgb_obj = 'reg:squarederror'
        else:
            qrf_criterion = 'absolute_error'
            # Deprecated
            xgb_obj = 'reg:linear'

        # QFR training
        self.logger.info('QRF training started')
        qrf_reg = qfrfQuantileRandomForestRegressor(n_estimators=lgb_reg.params['num_iterations'],
                                                    max_leaf_nodes=lgb_reg.params['num_leaves'],
                                                    max_features=lgb_reg.params['colsample_bytree'],
                                                    criterion=qrf_criterion)
        qrf_reg.fit(train_data_X, train_data_y, sample_weight=sample_weights)
        self.logger.info('QRF training ended')

        # NGB + XGB training
        self.logger.info('NGBoost training started')
        ngb_reg = NGBRegressor(n_estimators=lgb_reg.params['num_iterations'],
                               learning_rate=lgb_reg.params['learning_rate'], Dist=Normal, Base=default_tree_learner,
                               natural_gradient=True, verbose=False, Score=MLE, random_state=500)
        ngb_reg.fit(train_data_X, train_data_y, sample_weight=sample_weights)
        self.logger.info('NGBoost training ended')

        self.logger.info('XGBoost training started')
        xgb_reg = xgb.XGBRegressor(objective=xgb_obj, colsample_bytree=lgb_reg.params['colsample_bytree'],
                                   learning_rate=lgb_reg.params['learning_rate'], max_depth=5, alpha=10,
                                   n_estimators=lgb_reg.params['num_iterations'])
        xgb_reg.fit(train_data_X, train_data_y, sample_weight=sample_weights)
        self.logger.info('XGBoost training ended')

        self.saving_models_training_results(lgb_reg, qrf_reg, ngb_reg, xgb_reg)

    def saving_models_training_results(self, lgb_reg, qrf_reg, ngb_reg, xgb_reg):
        target_folder = '%smt' % self.result_folder
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self.logger.info('Save MT results in folder %s' % target_folder)

        # Inputs/features file
        _, output_signal, day_case = self.target.split('__')
        files_id = '%s-%s_%s' % (output_signal[1:], day_case, self.cfg['family'])

        with open('%s%sinputs_%s.json' % (target_folder, os.sep, files_id), 'w') as of:
            json.dump({'signals': list(self.df_X_best.columns)}, of, indent=2)

        # Metadata file
        ngboost_keys = ['natural_gradient', 'n_estimators', 'learning_rate', 'minibatch_frac', 'col_sample', 'verbose']
        ngboost_pars = {key: ngb_reg.get_params()[key] for key in ngboost_keys if key in ngb_reg.get_params()}
        metadata = {
            'general': {
                'case': self.forecast_type,
                'target': self.target,
                'family': self.cfg['family'],
                'region': self.region,
                'dataToConsiderMinLimit': self.cfg['regions'][self.region]['dataToConsiderMinLimit'],
                'datasetPeriod': self.cfg['dataset'],
                'weights': self.cfg['regions'][self.region]['weights'][self.forecast_type],
                'hpoBeforeFSPars': self.cfg['hpoBeforeFS'],
                'fsPars': self.cfg['fs'],
                'hpoAfterFSPars': self.cfg['hpoAfterFS'],
                'mtPars': self.cfg['mt']
            },
            "modelParameters": {
                "lightGBM": lgb_reg.params,
                "quantileRandomForestRegressor": qrf_reg.forest.get_params(),
                "ngboost": ngboost_pars,
                "xgboost": xgb_reg.get_params()
            },
            'stations': {
                'measureStations': self.cfg['regions'][self.region]['measureStations'],
                'forecastStations': self.cfg['regions'][self.region]['forecastStations'],
                'copernicusStations': self.cfg['regions'][self.region]['copernicusStations']
            },
        }
        with open('%s%smetadata_%s.json' % (target_folder, os.sep, files_id), 'w') as of:
            json.dump(metadata, of, indent=2)

        # Models file (structure needed for the compatibility with forecast_manager script
        pickle.dump([ngb_reg, qrf_reg, qrf_reg, xgb_reg, lgb_reg], open('%s%spredictor_%s.pkl' % (target_folder, os.sep,
                                                                                                  files_id), 'wb'))

    def corr_analysis(self, target):
        # Get features data
        feat_file_name = '%sfs/%s_%s_features_rank.csv' % (self.result_folder, self.cfg['family'], target)
        fs_data = pd.read_csv(feat_file_name)

        corrs = {}
        too_correlated = {}
        limit = self.cfg['fs']['corrAnalysis']['firstFeaturesToCheck']
        corr_threshold = self.cfg['fs']['corrAnalysis']['corrMaximumValue']
        features_to_consider = self.cfg['fs']['corrAnalysis']['featuresToSelect']

        for i in range(0, len(fs_data['feature'].values[0:limit])):
            x1_name = fs_data['feature'].values[i]
            self.logger.info('corr[(rank %.3d - %s) vs first %3i features] > %.2f: Start analysis' % (i, x1_name, limit,
                                                                                                      corr_threshold))
            flag_found = False
            for j in range(0, len(fs_data['feature'].values[0:limit])):
                x2_name = fs_data['feature'].values[j]
                if x1_name != x2_name and (x2_name, x1_name) not in corrs.keys():
                    series_x1 = self.df_X_all[x1_name]
                    series_x2 = self.df_X_all[x2_name]
                    corr_x1_x2 = series_x1.corr(series_x2)
                    corrs[(x1_name, x2_name)] = corr_x1_x2

                    if corr_x1_x2 > corr_threshold:
                        if x2_name not in too_correlated:
                            too_correlated[x2_name] = {'corr_with': x1_name, 'corr': corr_x1_x2}
                            flag_found = True

            if flag_found is True:
                self.logger.info('corr[(rank %.3d - %s) vs first %3i features] > %.2f: At least one case' % (i, x1_name,
                                                                                                             limit,
                                                                                                             corr_threshold))
            else:
                self.logger.info('corr[(rank %.3d - %s)) vs first %3i features] > %.2f: No cases' % (i, x1_name, limit,
                                                                                                     corr_threshold))

        # Add the correlations to the related columns
        res = []
        fs_data_corr = copy.deepcopy(fs_data.head(limit))
        for feat in fs_data['feature'].values[0:limit]:
            if feat in too_correlated.keys():
                res.append(too_correlated[feat])
            else:
                res.append(None)
        fs_data_corr['corr'] = res

        # Create the dataset without the too-correlated features
        fs_data_corr_filtered = fs_data_corr[fs_data_corr['corr'].isna() | (fs_data_corr['corr'] == None)]
        fs_data_corr_filtered = fs_data_corr_filtered.reset_index(drop=True)
        fs_data_corr_filtered['rank'] = fs_data_corr_filtered.index + 1
        fs_data_corr_filtered = fs_data_corr_filtered.drop(columns=['corr'])

        # Set the dataframe containing only the filtered features
        if len(fs_data_corr_filtered['feature'].values) < features_to_consider:
            self.logger.warning('Only %i passed the correlation test, they should be at least %i' %
                                (len(fs_data_corr_filtered['feature'].values), features_to_consider))
            self.df_X_best = self.df_X_all[fs_data_corr_filtered['feature'].values].copy()
        else:
            self.df_X_best = self.df_X_all[fs_data_corr_filtered['feature'].values[0:features_to_consider]].copy()

        # Save the results
        os.mkdir('%sfs_corr_filter' % self.result_folder)
        fs_data_corr_filtered.to_csv('%sfs_corr_filter/%s_%s_features_rank.csv' % (self.result_folder, self.cfg['family'], target), index=False)
        fs_data_corr.to_csv('%sfs_corr_filter/%s_%s_features_rank_all.csv' % (self.result_folder, self.cfg['family'], target), index=False)
        with open('%sfs_corr_filter/%s_%s_best_features.json' % (self.result_folder, self.cfg['family'], target), 'w') as of:
            json.dump({'signals': list(self.df_X_best.columns)}, of, indent=2)
