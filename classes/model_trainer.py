import numpy as np
import pandas as pd
import pickle
import os
import json
import glob
import scipy
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE, LogScore
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix

from classes.qrfr import QuantileRandomForestRegressor as qfrfQuantileRandomForestRegressor


class ModelTrainer:
    """
    This class will perform training on datasets. This could happen during the grid search for the best combination
    of weights, or, once the weights are assessed, for the creation of the final models
    """

    def __init__(self, features_analyzer, input_gatherer, forecast_type, cfg, logger):
        """
        Constructor

        :param features_analyzer: Features Analyzer
        :type features_analyzer: FeaturesAnalyzer
        :param input_gatherer: Inputs Gatherer
        :type input_gatherer: InputsGatherer
        :param forecast_type: Forecast type (MOR | EVE)
        :type forecast_type: str
        :param cfg: FTP parameters for the files exchange
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        """
        # set the variables
        self.features_analyzer = features_analyzer
        self.forecast_type = forecast_type
        self.input_gatherer = input_gatherer
        self.cfg = cfg
        self.logger = logger
        self.dataFrames = None
        self.models = {}

    def get_datasets(self):
        self.dataFrames = self.features_analyzer.dataFrames

    def get_accuracy_threshold(self, threshold, prediction, measured):
        """
        Calculate accuracy of predictions whose measured value is above a certain threshold

        :param threshold: threshold level
        :type threshold: float
        :param prediction: predicted data
        :type prediction: numpy.array
        :param measured: measured data
        :type measured: numpy.array
        :return: accuracy score
        :rtype: float
        """

        lcl_acc = 0.0
        if not measured.loc[measured > threshold].empty:
            lcl_acc = accuracy_score(self.get_classes(prediction.loc[measured > threshold]),
                                     self.get_classes(measured.loc[measured > threshold]))
        return lcl_acc

    @staticmethod
    def calc_mae_rmse_threshold(meas, pred, th):
        mask = meas.values >= th
        return round(mean_absolute_error(meas[mask], pred[mask]), 3), \
               round(np.sqrt(mean_squared_error(meas[mask], pred[mask])), 3)

    def calculate_KPIs(self, region, prediction, measured, weights):
        """
        For each fold and/or train/test separation, return the KPIs to establish the best weights combination

        :param prediction: predicted data
        :type prediction: numpy.array
        :param measured: measured data
        :type measured: numpy.array
        :return: pandas DF with KPIs for each dataset provided
        :rtype: pandas.DataFrame
        """
        threshold1 = self.cfg['regions'][region]['featuresAnalyzer']['threshold1']
        threshold2 = self.cfg['regions'][region]['featuresAnalyzer']['threshold2']
        threshold3 = self.cfg['regions'][region]['featuresAnalyzer']['threshold3']
        w1 = weights['w1']
        w2 = weights['w2']
        w3 = weights['w3']

        lcl_acc_1 = round(self.get_accuracy_threshold(threshold1, prediction, measured), 3)
        lcl_acc_2 = round(self.get_accuracy_threshold(threshold2, prediction, measured), 3)
        lcl_acc_3 = round(self.get_accuracy_threshold(threshold3, prediction, measured), 3)

        lcl_acc = round(accuracy_score(self.get_classes(prediction), self.get_classes(measured)), 3)
        lcl_rmse = round((mean_squared_error(measured, prediction) ** 0.5), 3)
        lcl_mae = round(mean_absolute_error(measured, prediction), 3)
        lcl_cm = confusion_matrix(self.get_classes(prediction), self.get_classes(measured))

        mae1, rmse1 = self.calc_mae_rmse_threshold(meas=measured, pred=prediction, th=threshold1)
        mae2, rmse2= self.calc_mae_rmse_threshold(meas=measured, pred=prediction, th=threshold2)
        mae3, rmse3= self.calc_mae_rmse_threshold(meas=measured, pred=prediction, th=threshold3)

        df_KPIs = pd.DataFrame([[w1, w2, w3, lcl_acc_1, lcl_acc_2, lcl_acc_3, lcl_acc, rmse1, rmse2, rmse3, lcl_rmse,
                                 mae1, mae2, mae3, lcl_mae, str(lcl_cm.flatten().tolist())]],
                               columns=['w1', 'w2', 'w3', 'Accuracy_1', 'Accuracy_2', 'Accuracy_3', 'Accuracy',
                                        'RMSE1', 'RMSE2', 'RMSE3', 'RMSE', 'MAE1', 'MAE2', 'MAE3', 'MAE', 'ConfMat'])

        return df_KPIs

    def get_numpy_df(self, df_x, df_y):
        x_data_no_date = df_x.iloc[:, 1:]
        y_data_no_date = df_y.iloc[:, 1:]

        assert (len(x_data_no_date) == len(y_data_no_date))
        x_data = np.array(x_data_no_date, dtype='float64')
        y_data = np.array(y_data_no_date, dtype='float64')
        return x_data, y_data

    def remove_date(self, X, Y):

        assert 'date' in X.columns.values
        assert 'date' in Y.columns.values
        X = X.iloc[:, 1:]
        Y = Y.iloc[:, 1:]
        assert 'date' not in X.columns.values
        assert 'date' not in Y.columns.values

        return X, Y

    def convert_to_series(self, prediction, Y):
        """
        Convert dataframes to series for easier KPIs calculation
        """

        assert (len(prediction) == len(Y))

        prediction = pd.Series(prediction, index=Y.index)
        measured = pd.Series(Y.iloc[:, 0], index=Y.index)

        return prediction, measured

    @staticmethod
    def calc_prob_interval(pred_dataset, lower_limit, upper_limit):
        mask = np.logical_and(pred_dataset > lower_limit, pred_dataset < upper_limit)
        return len(pred_dataset[mask]) / len(pred_dataset)

    @staticmethod
    def handle_qrf_output(cfg, qrf, input_vals, region_code):
        qntls = np.array(cfg['regions'][region_code]['forecaster']['quantiles'])
        pred_qntls, pred_dataset = qrf.predict(input_vals, qntls)
        pred_dataset = pred_dataset[0]
        pred_qntls = pred_qntls[0]

        ths = cfg['regions'][region_code]['forecaster']['thresholds']
        eps = np.finfo(np.float32).eps
        dict_probs = {'thresholds': {}, 'quantiles': {}}

        # Get probabilities to be in configured thresholds
        for i in range(1, len(ths)):
            dict_probs['thresholds']['[%i:%i]' % (ths[i-1], ths[i])] = ModelTrainer.calc_prob_interval(pred_dataset, ths[i-1], ths[i]-eps)
        dict_probs['thresholds']['[%i:%f]' % (ths[i], np.inf)] = ModelTrainer.calc_prob_interval(pred_dataset, ths[i], np.inf)

        # Get probabilities to be in the configured quantiles
        for i in range(0, len(qntls)):
            dict_probs['quantiles']['perc%.0f' % (qntls[i]*100)] = pred_qntls[i]

        return dict_probs

    @staticmethod
    def handle_ngb_normal_dist_output(cfg, mu, sigma, region_code):
        dist = scipy.stats.norm(loc=mu, scale=sigma)
        # QUANTILES
        # dist.ppf(0.1)
        # dist.ppf([0.1, 0.5])
        # VALUES FOR PROB
        # dist.ppf(0.1)
        # dist.ppf([0.1, 0.5])
        samples = []
        for i in range(1, 1000):
            samples.append(dist.ppf(float(i / 1000)))
        samples = np.array(samples)

        ths = cfg['regions'][region_code]['forecaster']['thresholds']
        eps = np.finfo(np.float32).eps
        dict_probs = {'thresholds': {}, 'quantiles': {}}

        for i in range(1, len(ths)):
            dict_probs['thresholds']['[%i:%i]' % (ths[i-1], ths[i])] = ModelTrainer.calc_prob_interval(samples, ths[i-1], ths[i]-eps)
        dict_probs['thresholds']['[%i:%f]' % (ths[i], np.inf)] = ModelTrainer.calc_prob_interval(samples, ths[i], np.inf)

        # Get probabilities to be in the configured quantiles
        for q in cfg['regions'][region_code]['forecaster']['quantiles']:
            dict_probs['quantiles']['perc%.0f' % (q*100)] = dist.ppf(q)

        return dict_probs

    def fold_training(self, region, train_index, test_index, X, Y, weights):
        """
        For each fold and/or tran/test separation, create the model and calculate KPIs to establish the best weights
        combination

        :param train_index: indexes of dataset that compose train set
        :type train_index: pandas.Index
        :param test_index: indexes of dataset that compose test set
        :type test_index: pandas.Index
        :param X: design matrix
        :type X: pandas.DataFrame
        :param Y: response vector
        :type Y: pandas.DataFrame
        :return: prediction performed on test dataset
        :rtype: numpy.array
        """

        Xtrain, Xtest = np.array(X.loc[train_index, :]), np.array(X.loc[test_index, :])
        Ytrain, Ytest = Y.loc[train_index].reset_index(drop=True), Y.loc[test_index].reset_index(drop=True)

        assert len(Xtrain) == len(Ytrain)
        assert len(Xtest) == len(Ytest)

        ngb  = self.train_NGB_model(region, Xtrain, Ytrain, weights)[0]

        return ngb.predict(Xtest)

    def train_NGB_model(self, region, Xtrain, Ytrain, weights, ngbPars=None):
        """
        Return the NGB model trained on the available data

        :param Xtrain: indexes of dataset that compose train set
        :type Xtrain: np.array()
        :param Ytrain: indexes of dataset that compose test set
        :type Ytrain: pandas.DataFrame
        :return: prediction model
        :rtype: ngboost.NGBRegressor
        """

        if ngbPars is None:
            # Usage of the configured parameters
            n_est = self.cfg['regions'][region]['featuresAnalyzer']['numberEstimatorsNGB']
            l_rate = self.cfg['regions'][region]['featuresAnalyzer']['learningRate']
        else:
            # Usage of the parameters passed as arguments
            n_est = ngbPars['numberEstimators']
            l_rate = ngbPars['learningRate']

        threshold1 = self.cfg['regions'][region]['featuresAnalyzer']['threshold1']  # 240
        threshold2 = self.cfg['regions'][region]['featuresAnalyzer']['threshold2']  # 180
        threshold3 = self.cfg['regions'][region]['featuresAnalyzer']['threshold3']  # 135
        w1 = weights['w1']
        w2 = weights['w2']
        w3 = weights['w3']

        weight = np.array(
            [w1 if x >= threshold1 else w2 if x >= threshold2 else w3 if x >= threshold3 else 1.0 for x in
             np.array(Ytrain)],
            dtype='float64')

        assert len(weight) == len(Ytrain)

        ngb = NGBRegressor(n_estimators=n_est, learning_rate=l_rate, Dist=Normal,
                           Base=default_tree_learner, natural_gradient=True, verbose=False,
                           Score=MLE, random_state=500).fit(Xtrain, np.array(Ytrain).ravel(), sample_weight=weight)

        return ngb, weight

    def error_data(self, pred, Y, fold, weights):
        """
        Create pandas df with weights, fold, measurements and predictions

        :param pred: predicted data
        :type pred: numpy.array
        :param Y: measured data
        :type Y: pandas.Series
        :param fold: current fold of Cross Validation
        :type fold: int
        :return: pandas DF with information
        :rtype: pandas.DataFrame
        """

        Y = np.array(Y.values)
        assert len(pred) == len(Y)

        df_pred = pd.DataFrame()
        df_pred['w1'] = [weights['w1']] * len(Y)
        df_pred['w2'] = [weights['w2']] * len(Y)
        df_pred['w3'] = [weights['w3']] * len(Y)
        df_pred['Fold'] = [fold] * len(Y)
        df_pred['Measurements'] = Y
        df_pred['Prediction'] = pred

        return df_pred

    def get_weights_folder_results(self, region, target_column, weights):
        root_output_folder_path = self.input_gatherer.output_folder_creator(region)
        str_ws = ''
        for kw in weights.keys():
            str_ws = '%s%s-%s_' % (str_ws, kw, weights[kw])
        str_ws = str_ws[0:-1]

        if not os.path.exists(root_output_folder_path + 'gs'):
            os.mkdir(root_output_folder_path + 'gs')

        if not os.path.exists(root_output_folder_path + 'gs' + os.sep + target_column):
            os.mkdir(root_output_folder_path + 'gs' + os.sep + target_column)

        if not os.path.exists(root_output_folder_path + 'gs' + os.sep + target_column + os.sep + str_ws):
            os.mkdir(root_output_folder_path + 'gs' + os.sep + target_column + os.sep + str_ws)

        return '%s%s%s%s%s%s%s' % (root_output_folder_path, 'gs', os.sep, target_column, os.sep, str_ws, os.sep)

    def training_cross_validated_fs(self, features, region, target_column, df_x, df_y, weights):
        df_x = df_x.reset_index(drop=True)
        df_y = df_y.reset_index(drop=True)

        ngb_prediction = np.empty(len(df_y))
        df_pred = pd.DataFrame(columns=['w1', 'w2', 'w3', 'Fold', 'Measurements', 'Prediction'])

        # Dataset preparation for CV
        df_x_tmp = df_x
        df_y_tmp = df_y
        df_x_tmp = df_x_tmp.drop(['date'], axis=1)
        df_y_tmp = df_y_tmp.drop(['date'], axis=1)
        cv_folds = self.cfg['regions'][region]['gridSearcher']['numFolds']
        kf = KFold(n_splits=cv_folds, shuffle=self.cfg['regions'][region]['gridSearcher']['shuffle'],
                   random_state=self.cfg['regions'][region]['gridSearcher']['randomState'])
        np_x = df_x_tmp.to_numpy()
        np_y = df_y_tmp.to_numpy()

        fold = 1
        for train_index, test_index in kf.split(np_x):
            # Get the I/O datasets for the training and the test
            X_train, X_test = np_x[train_index], np_x[test_index]
            y_train, y_test = np_y[train_index], np_y[test_index]

            # Perform the FS using the training folds
            self.logger.info('Region: %s, target: %s, weights: %s -> Started FS fold %i/%i' % (region,
                                                                                               target_column,
                                                                                               weights,
                                                                                               fold,
                                                                                               cv_folds))
            selected_features = self.features_analyzer.important_features(region, X_train, y_train, features[1:],
                                                                          weights)[0]
            X, Y = self.get_reduced_dataset(df_x, df_y, selected_features)
            X, Y = self.remove_date(X, Y)
            self.logger.info('Region: %s, target: %s, weights: %s -> Ended FS fold %i/%i' % (region, target_column,
                                                                                             weights, fold, cv_folds))

            # Perform the training using the training folds and the prediction with the test fold
            self.logger.info('Region: %s, target: %s, weights: %s -> Started model training fold %i/%i' % (region,
                                                                                                           target_column,
                                                                                                           weights,
                                                                                                           fold,
                                                                                                           cv_folds))
            pred = self.fold_training(region, train_index, test_index, X, Y, weights)
            ngb_prediction[test_index] = pred
            self.logger.info('Region: %s, target: %s, weights: %s -> Ended model training fold %i/%i' % (region,
                                                                                                         target_column,
                                                                                                         weights,
                                                                                                         fold,
                                                                                                         cv_folds))
            # Concat the prediction results
            df_pred = pd.concat([df_pred, self.error_data(pred, Y.loc[test_index], fold, weights)], ignore_index=True,
                                axis=0)
            fold += 1

        prediction, measured = self.convert_to_series(ngb_prediction, Y)
        return self.calculate_KPIs(region, prediction, measured, weights), df_pred

    def get_weights(self, input_file_name):
        w = {}
        str_w = ''
        for elem in input_file_name.split(os.sep)[-2].split('_'):
            code, val = elem.split('-')
            w[code] = int(val)
            str_w += val + '-'
        return w, str_w[:-1]

    def train_final_models(self, k_region, target_signal, hps=None):
        """
        Calculates the KPIs for a set of weight with multiple Feature selection: First we create the folds of the cross
        validation, then for each fold we do the feature selection and locally calculate the KPIs
        """
        target_data = self.cfg['regions'][k_region]['finalModelCreator']['targets'][target_signal]

        self.get_datasets()

        key = k_region
        df = self.dataFrames[key]

        fp = self.input_gatherer.output_folder_creator(key)

        _, _, _, df_x, df_y = self.features_analyzer.dataset_splitter(key, df, target_signal)

        # Check if there is a hyperparameters optimization or not
        if hps is None:
            suffix = self.cfg['regions'][k_region]['finalModelCreator']['signalsFileSuffix']
            input_files = glob.glob('%s*%s%s.json' % (fp, target_signal, suffix))
        else:
            str_hpars = 'ne%i-lr%s' % (hps['numberEstimators'], str(hps['learningRate']).replace('.', ''))
            suffix = str_hpars
            input_files = glob.glob('%shpo%s%s%s*%s*.json' % (fp, os.sep, suffix, os.sep, target_signal))

        for input_file in input_files:
            selected_features = json.loads(open(input_file).read())['signals']
            X, Y = self.get_reduced_dataset(df_x, df_y, selected_features)
            X, Y = self.remove_date(X, Y)

            target_data['weights'] = self.cfg['regions'][k_region]['featuresAnalyzer']['targetColumns'][target_signal]['weights'][self.forecast_type]

            start_year = self.cfg['datasetSettings']['years'][0]
            end_year = self.cfg['datasetSettings']['years'][-1]
            self.logger.info('Train models for %s - %s; period [%s:%s], case %s, weights: %s' % (k_region,
                                                                                                 target_signal,
                                                                                                 start_year,
                                                                                                 end_year,
                                                                                                 self.forecast_type,
                                                                                                 target_data['weights']))

            # Train NGB model
            self.logger.info('Target %s -> NGBoost model training start' % target_signal)
            ngb, weight = self.train_NGB_model(k_region, X, Y, target_data['weights'], hps)
            self.logger.info('Target %s -> NGBoost model training end' % target_signal)

            # Train QRF model
            rfqr = None
            # self.logger.info('RFQR model training start')
            # rfqr = RandomForestQuantileRegressor(n_estimators=1000).fit(X, np.array(Y).ravel())
            # self.logger.info('RFQR model training end')

            self.logger.info('Target %s -> pyquantrf RFQR model training start' % target_signal)
            rfqr_w = qfrfQuantileRandomForestRegressor(nthreads=4, n_estimators=1000, min_samples_leaf=10)
            rfqr_w.fit(X, np.array(Y).ravel(), sample_weight=weight)
            self.logger.info('Target %s -> pyquantrf RFQR model training end' % target_signal)

            # Check if there is a hyperparameters optimization or not
            if hps is None:
                file_name_noext = fp + 'predictor_' + target_data['label'] + '_' + \
                                  self.cfg['regions'][k_region]['finalModelCreator']['identifier']
            else:
                file_name_noext = '%shpo%s%s%spredictor_%s_%s' % (fp, os.sep, suffix, os.sep,target_data['label'],
                                                                  str_hpars)

            pickle.dump([ngb, rfqr, rfqr_w], open('%s.pkl' % file_name_noext, 'wb'))
            json.dump({"signals": list(selected_features)}, open('%s.json' % file_name_noext.replace('predictor', 'inputs'), 'w'))

    @staticmethod
    def get_reduced_dataset(df_x, df_y, selected_features):
        """
        Extract a smaller dataframe with the selected features as columns. Keep the date and refresh indices


        :param selected_features: list of selected features
        :type selected_features: list
        :param df_x: design matrix
        :type df_x: pandas.DataFrame
        :param df_y: response vector
        :type df_y: pandas.DataFrame
        :return: pandas DF with reduced columns
        :rtype: pandas.DataFrame, pandas.DataFrame
        """

        lcl_df_x = df_x.loc[:, ['date'] + selected_features]
        lcl_df_y = df_y

        # Date must be there
        assert len(lcl_df_x.columns.values) == len(selected_features) + 1
        assert len(lcl_df_y.columns.values) == 2
        assert len(lcl_df_y) == len(lcl_df_x)

        lcl_df_x = lcl_df_x.reset_index(drop=True)
        lcl_df_y = lcl_df_y.reset_index(drop=True)

        return lcl_df_x, lcl_df_y

    def get_classes(self, prediction):
        y_classes = []

        for element in prediction:
            if element < 60:
                y_classes.append(0)
            elif element < 120:
                y_classes.append(1)
            elif element < 135:
                y_classes.append(2)
            elif element < 180:
                y_classes.append(3)
            elif element < 240:
                y_classes.append(4)
            else:
                y_classes.append(5)

        return y_classes
