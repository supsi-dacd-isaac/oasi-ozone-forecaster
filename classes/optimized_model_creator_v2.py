import copy
import json
import os
import shap
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score

from pyforecaster.trainer import hyperpar_optimizer, retrieve_cv_results
from pyforecaster.metrics import err, squerr, rmse, nmae, mape


class OptimizedModelCreatorV2:
    """
    This class will perform hyperparameters optimization + feature selection + model training
    """
    def __init__(self, location, opt_cfg, main_cfg, logger):
        self.location = location
        self.predictor_cfg = opt_cfg
        self.main_cfg = main_cfg
        self.logger = logger

        self.sig_labels = []
        self.targets_labels = None
        self.x_all = None
        self.x_fs = None
        self.y_all = None
        self.selected_features = None

        self.features_importance = None
        self.mean_fs_score = None

        self.opt_pars = None
        self.opt_score = None

        self.lgb_regressors = None
        self.lgb_pars = None

        self.METRICS = {'nmae': nmae, 'err': err, 'squerr': squerr, 'rmse': rmse, 'mape': mape}

    def io_dataset_creation(self, raw_df):
        self.logger.info('Number of raw observations: %i' % len(raw_df))

        # We suppose that before 14:00 the forecast of the 2nd MeteoSuisse run,
        # operated around 12:00, is not available
        mask_1run = raw_df.index.hour <= 13

        self.sig_labels = []
        # Input signals handling (measures single values)
        input_meas_singleton_cfg = self.predictor_cfg['input']['measures']['singletons']
        for k_code in input_meas_singleton_cfg.keys():
            for i in range(0, len(input_meas_singleton_cfg[k_code]['backHours'])):
                in_code = k_code
                in_back_steps = input_meas_singleton_cfg[k_code]['backHours'][i]
                in_label = '%s__sb%02i' % (in_code, in_back_steps)

                if in_back_steps > 0:
                    raw_df.loc[:, in_label] = raw_df[in_code].shift(in_back_steps)
                else:
                    raw_df.loc[:, in_label] = raw_df[in_code]

                # Append the input code in the list
                self.sig_labels.append(in_label)

            # Dataframe defragmentation
            raw_df = raw_df.copy()

        # Input signals handling (aggregations of measures)
        input_meas_agg_cfg = self.predictor_cfg['input']['measures']['aggregations']
        for k_code in input_meas_agg_cfg.keys():
            for i in range(0, len(input_meas_agg_cfg[k_code]['backDays'])):
                step = input_meas_agg_cfg[k_code]['backDays'][i]
                func = k_code.split('func_')[-1]
                k_code_final = k_code.split('__func')[0]

                agg_sig_code = '%s__agg_db%i_%s' % (k_code_final, step, func)

                if func == 'mean':
                    agg_series = raw_df[k_code_final].resample('D').mean()
                elif func == 'max':
                    agg_series = raw_df[k_code_final].resample('D').max()
                elif func == 'min':
                    agg_series = raw_df[k_code_final].resample('D').min()
                elif func == 'std':
                    agg_series = raw_df[k_code_final].resample('D').stdev()
                agg_series_shifted = agg_series.shift(step)
                agg_df = pd.DataFrame({agg_sig_code: agg_series_shifted})
                raw_df[agg_sig_code] = raw_df.index.normalize().map(agg_df[agg_sig_code])

                # Append the input code in the list
                self.sig_labels.append(agg_sig_code)

            # Dataframe defragmentation
            raw_df = raw_df.copy()

        # Input signals handling (forecast single values)
        input_for_singleton_cfg = self.predictor_cfg['input']['forecast']['singletons']
        for k_code in input_for_singleton_cfg.keys():
            for i in range(0, len(input_for_singleton_cfg[k_code]['forwardHours'])):
                in_code = k_code
                in_forward_steps = self.predictor_cfg['input']['forecast']['singletons'][k_code]['forwardHours'][i]
                in_label = '%s__sf%02i' % (in_code, in_forward_steps)
                # We suppose that before 14:00 the forecast of the 2nd MeteoSuisse prediction running,
                # operated around 12:00, is not available
                raw_df.loc[mask_1run, in_label] = raw_df.loc[mask_1run, '%s_1run' % in_code]
                raw_df.loc[~mask_1run, in_label] = raw_df.loc[~mask_1run, '%s_2run' % in_code]

                # Append the input code in the list
                self.sig_labels.append(in_label)

            # Dataframe defragmentation
            raw_df = raw_df.copy()

        # Input signals handling (aggregations of forecast)
        input_for_agg_cfg = self.predictor_cfg['input']['forecast']['aggregations']
        for k_code in input_for_agg_cfg.keys():
            for i in range(0, len(input_for_agg_cfg[k_code]['forwardDays'])):
                step = input_for_agg_cfg[k_code]['forwardDays'][i]
                func = k_code.split('func_')[-1]
                k_code_final = k_code.split('__func')[0]

                agg_sig_code = '%s__agg_df%i_%s' % (k_code_final, step, func)

                if func == 'mean':
                    agg_series_1run = raw_df['%s_1run' % k_code_final].resample('D').mean()
                    agg_series_2run = raw_df['%s_2run' % k_code_final].resample('D').mean()
                elif func == 'max':
                    agg_series_1run = raw_df['%s_1run' % k_code_final].resample('D').max()
                    agg_series_2run = raw_df['%s_2run' % k_code_final].resample('D').max()
                elif func == 'min':
                    agg_series_1run = raw_df['%s_1run' % k_code_final].resample('D').min()
                    agg_series_2run = raw_df['%s_2run' % k_code_final].resample('D').min()
                elif func == 'std':
                    agg_series_1run = raw_df['%s_1run' % k_code_final].resample('D').stdev()
                    agg_series_2run = raw_df['%s_2run' % k_code_final].resample('D').stdev()

                if step > 0:
                    agg_series_shifted_1run = agg_series_1run.shift(step)
                    agg_series_shifted_2run = agg_series_2run.shift(step)
                    agg_df_1run = pd.DataFrame({'%s_1run' % agg_sig_code: agg_series_shifted_1run})
                    agg_df_2run = pd.DataFrame({'%s_2run' % agg_sig_code: agg_series_shifted_2run})
                else:
                    agg_df_1run = pd.DataFrame({'%s_1run' % agg_sig_code: agg_series_1run})
                    agg_df_2run = pd.DataFrame({'%s_2run' % agg_sig_code: agg_series_2run})
                raw_df['%s_1run' % agg_sig_code] = raw_df.index.normalize().map(agg_df_1run['%s_1run' % agg_sig_code])
                raw_df['%s_2run' % agg_sig_code] = raw_df.index.normalize().map(agg_df_2run['%s_2run' % agg_sig_code])

                # We suppose that before 14:00 the forecast of the 2nd MeteoSuisse run,
                # operated around 12:00, is not available
                raw_df.loc[mask_1run, agg_sig_code] = raw_df.loc[mask_1run, '%s_1run' % agg_sig_code]
                raw_df.loc[~mask_1run, agg_sig_code] = raw_df.loc[~mask_1run, '%s_2run' % agg_sig_code]

                # Append the input code in the list
                self.sig_labels.append(agg_sig_code)

            # Dataframe defragmentation
            raw_df = raw_df.copy()

        # Check if the dataset is built for a prediction (you need only features X) or for
        # features selection/hyperparameter optimization/training (you need both features X and target y)
        if 'target' in self.predictor_cfg.keys():
            # Output signals (targets) handling
            out_code = self.predictor_cfg['target']['code']
            self.targets_labels = []
            for out_forward_steps in self.predictor_cfg['target']['forwardHours']:
                out_target_label = '%s_sf%02i__out' % (out_code, out_forward_steps)
                raw_df[out_target_label] = raw_df[out_code].shift(-out_forward_steps)
                self.targets_labels.append(out_target_label)
                self.sig_labels.append(out_target_label)

        # Columns selection
        df_result = raw_df[self.sig_labels].copy()
        df_result.dropna(inplace=True)

        # Save the Xy datasets
        if 'target' in self.predictor_cfg.keys():
            # Drop the targets from the X dataset
            self.x_all = df_result.drop(columns=self.targets_labels)
            self.y_all = df_result[self.targets_labels]
        else:
            self.x_all = df_result
            self.y_all = None


        self.logger.info('Number of observations: %i' % len(self.x_all))

    def order_dataset(self, new_order):
        self.x_all = self.x_all[new_order]

    # Create a function with a cycle over the target and fs will become something like fs_single_target
    def fs(self):

        self.x_fs = {}
        self.features_importance = {}
        self.selected_features = {}
        self.mean_fs_score = {}

        for target in self.y_all.columns:
            self.logger.info('Running FS of target %s' % target)
            y_target = self.y_all[target].to_frame()

            kf = KFold(n_splits=self.main_cfg['fs']['cv']['folds'], shuffle=self.main_cfg['fs']['cv']['shuffle'])

            # Model initialization
            lgb_model = lgb.LGBMRegressor()
            lgb_model.set_params(
                learning_rate=self.main_cfg['fs']['pars']['learning_rate'],
                n_estimators=self.main_cfg['fs']['pars']['n_estimators'],
                colsample_bytree=self.main_cfg['fs']['pars']['colsample_bytree'],
                num_leaves=self.main_cfg['fs']['pars']['num_leaves']
            )

            shap_values_list = []
            for train_index, test_index in kf.split(self.x_all):
                X_train, X_test = self.x_all.iloc[train_index], self.x_all.iloc[test_index]
                y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]

                # Train the model
                lgb_model.fit(X_train, y_train)

                # Calculate SHAP values
                explainer = shap.TreeExplainer(lgb_model)
                shap_values = explainer.shap_values(X_test)

                # Aggregate shap values
                shap_values_list.append(np.abs(shap_values).mean(axis=0))

            # Calculate mean shap values across the folds
            mean_shap_values = np.mean(shap_values_list, axis=0)
            feat_importance = pd.Series(mean_shap_values, index=self.x_all.columns)

            # Select the top features
            feat_importance = feat_importance.sort_values(ascending=False)
            self.selected_features[target] = list(feat_importance.index[0:self.main_cfg['fs']['featuresSelected']])
            self.features_importance[target] = feat_importance.reset_index()
            self.features_importance[target].columns = ['feature', 'importance']
            self.features_importance[target] = self.features_importance[target].set_index('feature')
            self.x_fs[target] = self.x_all[self.selected_features[target]]

            # Re-evaluate model with selected features
            selected_lgb_model = lgb.LGBMRegressor()
            selected_lgb_model.set_params(
                learning_rate=self.main_cfg['fs']['pars']['learning_rate'],
                n_estimators=self.main_cfg['fs']['pars']['n_estimators'],
                colsample_bytree=self.main_cfg['fs']['pars']['colsample_bytree'],
                num_leaves=self.main_cfg['fs']['pars']['num_leaves']
            )
            folds_scores = cross_val_score(selected_lgb_model, self.x_fs[target], y_target, cv=kf,
                                           scoring=self.main_cfg['fs']['score'])

            self.logger.info("Cross-validation, target %s, metric mean %s score: %.4f" % (target,
                                                                                          self.main_cfg['fs']['score'],
                                                                                          -folds_scores.mean()))
            self.mean_fs_score[target] = -folds_scores.mean()

    def train(self):
        self.lgb_regressors = {}
        self.lgb_pars = {}

        # Cycle over the targets
        for target in self.targets_labels:
            y_target = self.y_all[target].to_frame()

            self.logger.info('Training started, target %s' % target)
            pars = copy.deepcopy(self.main_cfg['mt']['fixedParams'])
            if target in self.opt_pars.keys():
                pars.update(self.opt_pars[target])

            self.lgb_pars[target] = copy.deepcopy(pars)
            num_boost_round = pars.pop('n_estimators')
            self.lgb_regressors[target] = lgb.train(pars, lgb.Dataset(self.x_fs[target], label=y_target),
                                                    num_boost_round=num_boost_round)
            self.logger.info('Training ended, target %s' % target)

    def optimize(self):
        self.opt_pars = {}
        self.opt_score = {}

        # Cycle over the targets
        for target in self.targets_labels:
            self.logger.info('Hyperparameter optimization started: location: %s, target: %s' % (self.location, target))

            y_target = self.y_all[target].to_frame()

            # CV folds creation
            cv_idxs = self.get_sequential_cv_idxs(len(self.x_fs[target].index), self.main_cfg['hpo']['cv']['folds'])
            cv_folds = (f for f in cv_idxs)

            study, replies = hyperpar_optimizer(self.x_fs[target], y_target, lgb.LGBMRegressor(),
                                                n_trials=self.main_cfg['hpo']['trials'],
                                                metric=self.METRICS[self.main_cfg['hpo']['metric']],
                                                cv=cv_folds, param_space_fun=self.param_space_fun,
                                                hpo_type=self.main_cfg['hpo']['cv']['type'],
                                                callbacks=[self.stop_on_no_improvement_callback])

            trials_df = retrieve_cv_results(study)
            assert trials_df['value'].isna().sum() == 0
            self.logger.info('Location: %s, target %s -> params: %s' % (self.location, target, study.best_params))
            self.logger.info('Location: %s, target %s -> score: %.4f' % (self.location, target, study.best_value))
            self.logger.info('Hyperparameter optimization ended: location: %s, target: %s' % (self.location, target))
            self.opt_pars[target] = study.best_params
            self.opt_score[target] = study.best_value


    def stop_on_no_improvement_callback(self, study, _):
        window = self.main_cfg['hpo']['noImprovementWindow']

        horizon = window
        if len(study.trials) > window:
            horizon += study.best_trial.number
            last_trials = study.trials[-(window + 1):]

            if last_trials[-1].number >= horizon:
                self.logger.warning(f"No improvement in last {window} trials, stopping the optimization.")
                study.stop()

    def param_space_fun(self, trial):
        par_space = {}
        for par_conf in self.main_cfg['hpo']['paramSpace']:
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

    def save(self):
        # Cycle over the targets
        for target in self.targets_labels:
            target_folder = '%s%s/%s/' % (self.main_cfg['outputFolder'], self.main_cfg['family'], target)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            self.logger.info('Save data in folder %s' % target_folder)

            # Main settings
            metadata = {
                'general': {
                    'family': self.main_cfg['family'],
                    'location': self.location,
                    'target': target,
                    'datasetPeriod': self.main_cfg['dataset'],
                    'fsPars': self.main_cfg['fs'],
                    'hpoPars': self.main_cfg['hpo'],
                    'mtPars': self.main_cfg['mt']
                },
                'predictorPars': self.predictor_cfg,
                'fsResults': { 'crossValidationMeanScore': self.mean_fs_score[target] },
                'optResults': { 'lightGBM': { 'score': self.opt_score[target], 'optimizedPars': self.lgb_pars[target]} }
            }
            with open('%s%s___%s___main_cfg.json' % (target_folder, self.main_cfg['family'], target), 'w') as of:
                json.dump(metadata, of, indent=2)

            # FS data saving
            self.features_importance[target].to_csv('%s%s___%s___ranks.csv' % (target_folder, self.main_cfg['family'],
                                                                               target))

            # Signals file
            with open('%s%s___%s___signals.json' % (target_folder, self.main_cfg['family'], target), 'w') as of:
                json.dump({'target': target, 'input': self.selected_features[target]}, of, indent=2)


            # Model file
            pickle.dump(self.lgb_regressors[target], open('%s%s___%s___predictor.pkl' % (target_folder, self.main_cfg['family'], target), 'wb'))


