import numpy as np
import pandas as pd
import pickle
import os
import json
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
from skgarden import RandomForestQuantileRegressor


class ModelTrainer:
    """This class will perform training on datasets. This could happen during the grid search for the best combination
    of weights, or, once the weights are assessed, for the creation of the final models"""

    def __init__(self, features_analyzer, input_gatherer, forecast_type, cfg, logger):
        """
        Constructor
        :param inputs_gatherer: Inputs Gatherer
        :type inputs_gatherer: InputsGatherer
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
        """Calculate accuracy of predictions whose measured value is above a certain threshold"""

        lcl_acc = 0.0
        if not measured.loc[measured > threshold].empty:
            lcl_acc = accuracy_score(self.get_classes(prediction.loc[measured > threshold]),
                                     self.get_classes(measured.loc[measured > threshold]))
        return lcl_acc

    def calculate_KPIs(self, prediction, measured):
        """For each fold and/or train/test separation, return the KPIs to establish the best weights combination"""

        threshold1 = self.cfg['featuresAnalyzer']['threshold1']
        threshold2 = self.cfg['featuresAnalyzer']['threshold2']
        threshold3 = self.cfg['featuresAnalyzer']['threshold3']
        w1 = self.cfg['featuresAnalyzer']['w1']
        w2 = self.cfg['featuresAnalyzer']['w2']
        w3 = self.cfg['featuresAnalyzer']['w3']

        lcl_acc_1 = round(self.get_accuracy_threshold(threshold1, prediction, measured), 3)
        lcl_acc_2 = round(self.get_accuracy_threshold(threshold2, prediction, measured), 3)
        lcl_acc_3 = round(self.get_accuracy_threshold(threshold3, prediction, measured), 3)

        lcl_acc = round(accuracy_score(self.get_classes(prediction), self.get_classes(measured)), 3)
        lcl_rmse = round((mean_squared_error(measured, prediction) ** 0.5), 3)
        lcl_mae = round(mean_absolute_error(measured, prediction), 3)
        lcl_cm = confusion_matrix(self.get_classes(prediction), self.get_classes(measured))

        df_KPIs = pd.DataFrame([[w1, w2, w3, lcl_acc_1, lcl_acc_2, lcl_acc_3, lcl_acc, lcl_rmse, lcl_mae,
                                 str(lcl_cm.flatten().tolist())]],
                               columns=['w1', 'w2', 'w3', 'Accuracy_1', 'Accuracy_2', 'Accuracy_3', 'Accuracy',
                                        'RMSE', 'MAE', 'ConfMat'])

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
        """Convert dataframes to series for easier KPIs calculation"""

        assert (len(prediction) == len(Y))

        prediction = pd.Series(prediction, index=Y.index)
        measured = pd.Series(Y.iloc[:, 0], index=Y.index)

        return prediction, measured

    def fold_training(self, train_index, test_index, X, Y):
        """For each fold and/or tran/test separation, create the model and calculate KPIs to establish the best weights
        combination"""

        Xtrain, Xtest = np.array(X.loc[train_index, :]), np.array(X.loc[test_index, :])
        Ytrain, Ytest = Y.loc[train_index].reset_index(drop=True), Y.loc[test_index].reset_index(drop=True)

        assert len(Xtrain) == len(Ytrain)
        assert len(Xtest) == len(Ytest)

        ngb = self.train_NGB_model(Xtrain, Ytrain)[0]

        return ngb.predict(Xtest)

    def train_NGB_model(self, Xtrain, Ytrain):
        """Return the NGB model trained on the available data"""

        n_est = self.cfg['featuresAnalyzer']['numberEstimatorsNGB']
        l_rate = self.cfg['featuresAnalyzer']['learningRate']
        threshold1 = self.cfg['featuresAnalyzer']['threshold1']  # 240
        threshold2 = self.cfg['featuresAnalyzer']['threshold2']  # 180
        threshold3 = self.cfg['featuresAnalyzer']['threshold3']  # 135
        w1 = self.cfg['featuresAnalyzer']['w1']
        w2 = self.cfg['featuresAnalyzer']['w2']
        w3 = self.cfg['featuresAnalyzer']['w3']

        weight = np.array(
            [w1 if x >= threshold1 else w2 if x >= threshold2 else w3 if x >= threshold3 else 0.1 for x in
             np.array(Ytrain)],
            dtype='float64')

        assert len(weight) == len(Ytrain)

        ngb = NGBRegressor(n_estimators=n_est, learning_rate=l_rate, Dist=Normal,
                           Base=default_tree_learner, natural_gradient=True, verbose=False,
                           Score=MLE, random_state=500).fit(Xtrain, np.array(Ytrain).ravel(), sample_weight=weight)

        return ngb, weight

    def get_KPIs_with_random_separation(self, features, df_x, df_y):
        """Normally used for testing only. This method uses 80% ot the dataset for training and the rest for testing"""

        x_data, y_data = self.get_numpy_df(df_x, df_y)
        selected_features = self.features_analyzer.important_features(x_data, y_data, features[1:])[0]
        X, Y = self.get_reduced_dataset(df_x, df_y, selected_features)

        limit = int(0.8 * len(X))
        test_index = X.loc[X.index >= limit, :].index
        train_index = X.loc[X.index < limit, :].index
        assert (len(test_index) + len(train_index) == len(X))

        X, Y = self.remove_date(X, Y)

        pred = self.fold_training(train_index, test_index, X, Y)
        prediction = pd.Series(pred, index=test_index)
        measured = pd.Series(Y.iloc[test_index, 0], index=test_index)

        return self.calculate_KPIs(prediction, measured), self.error_data(pred, Y.loc[test_index], 1)

    def error_data(self, pred, Y, fold):
        """Create pandas df with weights, fold, measurements and predictions"""

        Y = np.array(Y.values)
        assert len(pred) == len(Y)

        w1 = self.cfg['featuresAnalyzer']['w1']
        w2 = self.cfg['featuresAnalyzer']['w2']
        w3 = self.cfg['featuresAnalyzer']['w3']

        df_pred = pd.DataFrame()
        df_pred['w1'] = [w1] * len(Y)
        df_pred['w2'] = [w2] * len(Y)
        df_pred['w3'] = [w3] * len(Y)
        df_pred['Fold'] = [fold] * len(Y)
        df_pred['Measurements'] = Y
        df_pred['Prediction'] = pred

        return df_pred

    def training_cross_validated_single_FS(self, features, df_x, df_y):
        """This method calculates the KPIs for a set of weight with one single Feature selection: First we do the
        feature selection on the whole dataset, then we reduce and split it to calculate the KPIs on each fold"""

        x_data, y_data = self.get_numpy_df(df_x, df_y)
        selected_features = self.features_analyzer.important_features(x_data, y_data, features[1:])[0]
        X, Y = self.get_reduced_dataset(df_x, df_y, selected_features)

        cv_folds = []
        years = sorted(list(set(X.date.str[0:4])))

        for year in years:
            test_index = X.loc[X.date.str[0:4] == str(year), :].index
            train_index = X.loc[X.date.str[0:4] != str(year), :].index
            assert (len(test_index) + len(train_index) == len(X))
            cv_folds.append((train_index, test_index))

        assert (sum([len(fold[1]) for fold in cv_folds]) == len(X))

        X, Y = self.remove_date(X, Y)

        ngb_prediction = np.empty(len(Y))
        df_pred = pd.DataFrame(columns=['w1', 'w2', 'w3', 'Fold', 'Measurements', 'Prediction'])

        fold = 1
        for (train_index, test_index) in cv_folds:
            pred = self.fold_training(train_index, test_index, X, Y)
            ngb_prediction[test_index] = pred
            df_pred = pd.concat([df_pred, self.error_data(pred, Y.loc[test_index], fold)], ignore_index=True, axis=0)
            fold += 1

        prediction, measured = self.convert_to_series(ngb_prediction, Y)

        return self.calculate_KPIs(prediction, measured), df_pred

    def training_cross_validated_multiple_FS(self, features, df_x, df_y):
        """This method calculates the KPIs for a set of weight with multiple Feature selection: First we create the
        folds of the cross validation, then for each fold we do the feature selection and locally calculate the KPIs"""

        cv_folds = []
        years = sorted(list(set(df_x.date.str[0:4])))

        df_x = df_x.reset_index(drop=True)
        df_y = df_y.reset_index(drop=True)

        for year in years:
            test_index = df_x.loc[df_x.date.str[0:4] == str(year), :].index
            train_index = df_x.loc[df_x.date.str[0:4] != str(year), :].index
            assert (len(test_index) + len(train_index) == len(df_x))
            cv_folds.append((train_index, test_index))

        assert (sum([len(fold[1]) for fold in cv_folds]) == len(df_x))

        ngb_prediction = np.empty(len(df_y))
        df_pred = pd.DataFrame(columns=['w1', 'w2', 'w3', 'Fold', 'Measurements', 'Prediction'])
        fold = 1

        for (train_index, test_index) in cv_folds:
            lcl_x, lcl_y = df_x.loc[train_index], df_y.loc[train_index]
            x_data, y_data = self.get_numpy_df(lcl_x, lcl_y)
            selected_features = self.features_analyzer.important_features(x_data, y_data, features[1:])[0]
            X, Y = self.get_reduced_dataset(df_x, df_y, selected_features)
            X, Y = self.remove_date(X, Y)

            pred = self.fold_training(train_index, test_index, X, Y)
            ngb_prediction[test_index] = pred
            df_pred = pd.concat([df_pred, self.error_data(pred, Y.loc[test_index], fold)], ignore_index=True, axis=0)
            fold += 1

        prediction, measured = self.convert_to_series(ngb_prediction, Y)

        return self.calculate_KPIs(prediction, measured), df_pred

    def train_final_models(self):
        """This method calculates the KPIs for a set of weight with multiple Feature selection: First we create the
        folds of the cross validation, then for each fold we do the feature selection and locally calculate the KPIs"""

        self.get_datasets()

        for key, df in self.dataFrames.items():
            fp = self.input_gatherer.output_folder_creator(key)
            fn = fp + fp.split(os.sep)[1]

            x_data, y_data, features, df_x, df_y = self.features_analyzer.dataset_splitter(key, df)

            x_data, y_data = self.get_numpy_df(df_x, df_y)
            selected_features = self.features_analyzer.important_features(x_data, y_data, features[1:])[0]
            X, Y = self.get_reduced_dataset(df_x, df_y, selected_features)

            X, Y = self.remove_date(X, Y)

            ngb, weight = self.train_NGB_model(X, Y)

            rfqr = RandomForestQuantileRegressor(n_estimators=1000).fit(X, np.array(Y).ravel(), sample_weight=weight)
            rfqr_no_w = RandomForestQuantileRegressor(n_estimators=1000).fit(X, np.array(Y).ravel(), sample_weight=None)

            n_features = str(len(selected_features))

            pickle.dump([ngb, rfqr, rfqr_no_w], open(fn + '_mdl_' + n_features + '.pkl', 'wb'))
            pickle.dump(selected_features, open(fn + '_feats_' + n_features + '.pkl', 'wb'))
            json.dump({"signals": list(selected_features)}, open(fn + '_feats_' + n_features + '.json', 'w'))

    @staticmethod
    def get_reduced_dataset(df_x, df_y, selected_features):
        """Extract a smaller dataframe with the selected features as columns. Keep the date and refresh indices"""

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
