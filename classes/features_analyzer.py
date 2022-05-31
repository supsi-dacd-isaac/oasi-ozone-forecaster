import os

import numpy as np
import pandas as pd
import shap
import json

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE, LogScore

from classes.inputs_gatherer import InputsGatherer


class FeaturesAnalyzer:
    """
    Given a dataset composed of features on the columns and days on the rows of a pandas df, this class computes the
    best features and their importance
    """

    def __init__(self, inputs_gatherer, forecast_type, cfg, logger):
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
        self.inputs_gatherer = inputs_gatherer
        self.forecast_type = forecast_type
        self.cfg = cfg
        self.logger = logger
        self.dataFrames = None
        self.output_folder_name = None
        self.current_name = None
        self.nan_features = None

    def dataset_creator(self):
        """
        Build the datasets according to the instructions in the config file in the datasetSettings section
        """
        self.inputs_gatherer.dataframe_builder_regions()

    def update_datasets(self, name, output_dfs, target_columns):
        """
        Initialize folders and add metadata to container of datasets
        """

        folder_path = self.inputs_gatherer.output_folder_creator(name)
        file_path_df = folder_path + folder_path.split(os.sep)[1] + '_dataset.csv'
        if not os.path.isfile(file_path_df):
            self.logger.error('File %s does not exist' % file_path_df)
        tmp_df = pd.read_csv(file_path_df)

        # Filtering on data -> only observations related to output values higher than the limit will be considered
        mask = tmp_df[target_columns[0]] >= self.cfg['regions'][name]['dataToConsiderMinLimit']
        output_dfs[name] = {'dataset': tmp_df[mask], 'targetColumns': target_columns}

        # Select only configured input signals
        input_signals = self.inputs_gatherer.generate_input_signals_codes(name)
        candidate_signals = list(output_dfs[name]['dataset'].columns)
        # Remove date and output from candidates list
        candidate_signals.remove('date')
        for target_column in self.cfg['regions'][name]['targetColumns']:
            candidate_signals.remove(target_column)

        for candidate_signal in candidate_signals:
            if candidate_signal not in input_signals:
                # This signal has not to be used in the grid search
                output_dfs[name]['dataset'] = output_dfs[name]['dataset'].drop(candidate_signal, axis=1)

        return output_dfs

    def dataset_reader(self, region, target_column):
        """
        Read a previously created or provided csv file. If the dataset is created from a custom JSON or
        from regionals signals, this method has to be preceded by a call of dataset_creator
        """

        output_dfs = {}
        output_dfs = self.update_datasets(region, output_dfs, target_column)

        self.dataFrames = output_dfs

    def dataset_splitter(self, region, data, target_column):
        """
        Split a dataFrame in design matrix X and response vector Y

        :param name: code name of the region/json/csv
        :type name: str
        :param data: full dataset
        :type data: pandas.DataFrame
        :return: split datasets in multiple formats
        :rtype: numpy.array, numpy.array, list, pandas.DataFrame, pandas.DataFrame
        """
        # todo CHECK THIS PART (probably useless!)
        # self.current_name = name
        # df = data['dataset']

        # y_data = pd.DataFrame()
        # x_data = pd.DataFrame()
        # df_years = list(dict.fromkeys(df['date'].str[:4]))

        # # If we're at MOR the value of the max ozone of day ahead is our target. If we're at EVE, it is the max
        # # value of 2 days ahead
        # days_ahead = 1 if self.forecast_type == 'MOR' else 2
        #
        # for year in df_years:
        #     lcl_df = df.loc[df['date'].str[:4] == year, :].reset_index(drop=True)
        #     lcl_y_data = lcl_df.loc[days_ahead:, ['date', target_column]]
        #     lcl_x_data = lcl_df.iloc[:-days_ahead, :]
        #     y_data = pd.concat([y_data, lcl_y_data], axis=0).reset_index(drop=True)
        #     x_data = pd.concat([x_data, lcl_x_data], axis=0).reset_index(drop=True)
        # # Remove the target column
        # x_data = x_data.drop(target_column, axis=1)

        # Create the inputs dataset (x_data)
        x_data = data['dataset']
        # Drop from the input dataset all the output variables defined for this region in the dataset
        for target in self.cfg['regions'][region]['targetColumns']:
            x_data = x_data.drop(target, axis=1)

        # Create the outputs dataset (x_data)
        y_data = pd.DataFrame({'date': data['dataset']['date'], target_column: data['dataset'][target_column]})

        assert (len(x_data) == len(y_data))

        # Post processing of the downloaded/read data
        nan_rows = x_data.loc[x_data.isnull().any(axis=1), 'date']
        self.nan_features = x_data.loc[:, x_data.isnull().any()].columns.values

        if len(nan_rows) > 0:
            self.logger.warning("NaN found in the dataset in %i dates on %i (%.1f%%). The days related to the nan "
                                "will be removed from the dataset" % (len(nan_rows), len(x_data),
                                                                      len(nan_rows)/len(x_data)*1e2))
            # for row in nan_rows:
            #     self.debug.warning(row)

        x_data = x_data.drop(nan_rows.index, axis=0)
        x_data_no_date = x_data.iloc[:, 1:]
        y_data = y_data.drop(nan_rows.index, axis=0)
        y_data_no_date = y_data.iloc[:, 1:]

        assert (len(x_data_no_date) == len(y_data_no_date))

        features = x_data.columns.values
        features = features[features != target_column]
        x_data_np = np.array(x_data_no_date, dtype='float64')
        y_data_np = np.array(y_data_no_date, dtype='float64')

        return x_data_np, y_data_np, features, x_data, y_data

    def important_features(self, region, x_data, y_data, features, target_data, ngbPars=None):
        """
        Calculate the important features given design matrix, target vector and full list of features

        :param x_data: design matrix
        :type x_data: numpy.array
        :param y_data: response vector
        :type y_data: numpy.array
        :param features: list of features names
        :type features: list
        :return: list of new features and dataframe with relative importance of each single feature
        :rtype: list, pandas.DataFrame
        """

        assert x_data.shape[1] == len(features)

        if 'weights' in target_data.keys():
            # FS case
            weights = target_data['weights'][self.forecast_type]
        else:
            # HPOPT case
            weights = target_data

        if ngbPars is None:
            # Usage of the configured parameters
            n_est = target_data['numberEstimatorsNGB'][self.forecast_type]
            l_rate = target_data['learningRateNGB'][self.forecast_type]
        else:
            # Usage of the parameters passed as arguments
            n_est = ngbPars['numberEstimators']
            l_rate = ngbPars['learningRate']

        n_feat = self.cfg['regions'][region]['featuresAnalyzer']['numberSelectedFeatures']
        threshold1 = self.cfg['regions'][region]['featuresAnalyzer']['threshold1']
        threshold2 = self.cfg['regions'][region]['featuresAnalyzer']['threshold2']
        threshold3 = self.cfg['regions'][region]['featuresAnalyzer']['threshold3']

        w1 = weights['w1']
        w2 = weights['w2']
        w3 = weights['w3']

        NGB_model = NGBRegressor(learning_rate=l_rate, Base=default_tree_learner, Dist=Normal, Score=MLE,
                                 n_estimators=n_est, random_state=500, verbose=False)

        weights = np.array(
            [w1 if x >= threshold1 else w2 if x >= threshold2 else w3 if x >= threshold3 else 1.0 for x in y_data],
            dtype='float64')
        ngb = NGB_model.fit(x_data, y_data.ravel(), sample_weight=weights)
        explainer = shap.TreeExplainer(ngb, x_data, model_output=0)
        shap_values = explainer.shap_values(x_data, check_additivity=False)
        important_features = pd.DataFrame(list(zip(features, np.abs(shap_values).mean(0))),
                                          columns=['feature', 'feature_importance'])
        important_features = important_features.sort_values(by=['feature_importance'], ascending=False).reset_index(
            drop=True)
        new_features = list(important_features['feature'][:n_feat])

        return new_features, important_features

    def perform_feature_selection(self, region, x_data, y_data, features, target, target_data, hps=None):
        """
        Obtain selected features and also save them in the output folder

        :param x_data: design matrix
        :type x_data: numpy.array
        :param y_data: response vector
        :type y_data: numpy.array
        :param features: list of features names
        :type features: list
        :return: list of new features and dataframe with relative importance of each single feature
        :rtype: list, pandas.DataFrame
        """
        if hps is not None:
            self.logger.info('HPO STEP: %s' % hps)

        self.logger.info('Launched FS (%s variables to select, weights=[%s], samples=%i), '
                         'it can take a while...' % (self.cfg['regions'][region]['featuresAnalyzer']['numberSelectedFeatures'],
                                                     target_data['weights'][self.forecast_type], len(y_data)))
        new_features, important_features = self.important_features(region, x_data, y_data, features[1:],
                                                                   target_data, hps)


        important_nan_features = [f for f in self.nan_features if f in new_features]
        if len(important_nan_features) > 0:
            self.logger.warning(
                "The following %s features with missing data were found to be important and thus they should be filled in:" % str(
                    len(important_nan_features)))
            for f in important_nan_features:
                self.logger.warning(f)

        # Check if there is a hyperparameters optimization or not
        if hps is None:
            str_pars = None
            output_folder_path = self.inputs_gatherer.output_folder_creator(region)
        else:
            str_pars = 'ne%i-lr%s' % (hps['numberEstimators'], str(hps['learningRate']).replace('.', ''))
            output_folder_path = '%shpo%s%s%s' % (self.inputs_gatherer.output_folder_creator(region), os.sep, str_pars,
                                                  os.sep)
            if os.path.exists(output_folder_path) is False:
                os.mkdir(output_folder_path)

        # discard GLOB__step0 case
        clean_new_features = self.clean_features_list(region, important_features, new_features, str_pars)
        self.save_csv(important_features, target, clean_new_features, output_folder_path)

        return new_features, important_features

    def clean_features_list(self, region, important_features, new_features, str_pars):
        clean_new_features = []
        # Irradiance forecast at step 0 have not be considered
        for nf in new_features:
            if '__GLOB__step0' not in nf:
                clean_new_features.append(nf)
            else:
                if str_pars is not None:
                    self.logger.warning('%s %s -> skipped %s' % (region, str_pars, nf))

        for i in range(0, len(new_features) - len(clean_new_features)):
            clean_new_features.append(important_features['feature'][i+self.cfg['regions'][region]['featuresAnalyzer']['numberSelectedFeatures']])
        return clean_new_features

    def save_csv(self, important_features, target, new_features, output_folder_path):
        """
        Save selected features and their relative importance

        :param important_features: dataframe of the selected features and their relative importance
        :type important_features: pandas.DataFrame
        :param new_features: selected features
        :type new_features: list
        """
        fp = output_folder_path

        if not os.path.exists(fp):
            self.logger.error("Saving folder not found")

        output_df = pd.DataFrame(range(1, len(important_features) + 1), columns=['rank'])
        output_df = pd.concat([output_df, important_features], axis=1)
        output_df.to_csv(fp + fp.split(os.sep)[1] + '_' + target + '_features_importance.csv', index=False, header=True)

        fn = fp + fp.split(os.sep)[1] + '_' + target + '_signals.json'
        with open(fn, 'w') as f:
            json.dump({"signals": new_features}, f)
