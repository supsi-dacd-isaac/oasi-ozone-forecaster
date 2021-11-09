import numpy as np
import pandas as pd
import shap
import os
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE

from classes.inputs_gatherer import InputsGatherer


class FeaturesAnalyzer:
    """Given a dataset composed of features on the columns and days on the rows of a pandas df, this class computes the
    best features and their importance"""

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

    def dataset_creator(self):
        """This method builds the datasets according to the instructions in the config file in the datasetSettings
        section"""

        if self.cfg["featuresAnalyzer"]["datasetCreator"] == 'customJSON':
            self.inputs_gatherer.dataframe_builder_custom()
        elif self.cfg["featuresAnalyzer"]["datasetCreator"] == 'regions':
            self.inputs_gatherer.dataframe_builder_regions()
        elif self.cfg["featuresAnalyzer"]["datasetCreator"] == 'CSVreader':
            self.inputs_gatherer.dataframe_reader()
        else:
            self.logger.error('Option for dataset_creator is not valid. Available options are "customJSON", "regions" or "CSVreader"')

        self.dataFrames = self.inputs_gatherer.output_dfs

    def dataset_splitter(self, name, data):
        """Split a dataFrame in design matrix X and response vector Y"""

        self.current_name = name
        df = data['dataset']

        if len(data['targetColumns']) == 0:
            self.logger.error('Target column was not specified. Feature selection can not be performed')
        if len(data['targetColumns']) == 1:
            target_column = data['targetColumns'][0]
        else:
            target_column = 'Target'
            df[target_column] = df.loc[:, data['targetColumns']].max(axis=1)

        y_data = pd.DataFrame()
        x_data = pd.DataFrame()
        df_years = list(dict.fromkeys(df['date'].str[:4]))

        # If we're at MOR the value of the max ozone of day ahead is our target. If we're at EVE, it is the max
        # value of 2 days ahead
        days_ahead = 1 if self.forecast_type == 'MOR' else 2

        for year in df_years:
            lcl_df = df.loc[df['date'].str[:4] == year, :].reset_index(drop=True)
            lcl_y_data = lcl_df.loc[days_ahead:, target_column]
            lcl_x_data = lcl_df.iloc[:-days_ahead, 1:]
            y_data = pd.concat([y_data, lcl_y_data], axis=0).reset_index(drop=True)
            x_data = pd.concat([x_data, lcl_x_data], axis=0).reset_index(drop=True)

        assert (len(x_data) == len(y_data))

        # remove target column previously created, although it could make a nice feature
        if len(data['targetColumns']) > 1:
            x_data = x_data.drop(columns=[target_column])

        features = x_data.columns.values
        x_data = np.array(x_data, dtype='float64')
        y_data = np.array(y_data, dtype='float64')

        return x_data, y_data, features

    def perform_feature_selection(self, x_data, y_data, features):

        n_est = self.cfg['featuresAnalyzer']['numberEstimatorsNGB']
        l_rate = self.cfg['featuresAnalyzer']['learningRate']
        n_feat = self.cfg['featuresAnalyzer']['numberSelectedFeatures']
        w1 = self.cfg['featuresAnalyzer']['w1']
        w2 = self.cfg['featuresAnalyzer']['w2']
        w3 = self.cfg['featuresAnalyzer']['w3']
        threshold1 = self.cfg['featuresAnalyzer']['threshold1'] #240
        threshold2 = self.cfg['featuresAnalyzer']['threshold2'] #180
        threshold3 = self.cfg['featuresAnalyzer']['threshold3'] #135

        NGB_model = NGBRegressor(learning_rate=l_rate, Base=default_tree_learner, Dist=Normal, Score=MLE,
                                 n_estimators=n_est, random_state=500)
        weights = np.array(
            [w1 if x >= threshold1 else w2 if x >= threshold2 else w3 if x >= threshold3 else 0.1 for x in y_data], dtype='float64')
        ngb = NGB_model.fit(x_data, y_data.ravel(), sample_weight=weights)
        explainer = shap.TreeExplainer(ngb, x_data, model_output=0)
        shap_values = explainer.shap_values(x_data, check_additivity=False)
        important_features = pd.DataFrame(list(zip(features, np.abs(shap_values).mean(0))),
                                          columns=['feature', 'feature_importance'])
        important_features.sort_values(by=['feature_importance'], ascending=False, inplace=True)
        new_features = list(important_features['feature'][:n_feat])

        self.save_csv(important_features)

        return new_features, important_features

    def save_csv(self, important_features):
        folder_path = self.inputs_gatherer.output_folder_creator(self.current_name)

        if not os.path.exists(folder_path):
            self.logger.error("Saving folder not found")

        output_df = pd.DataFrame(range(1, len(important_features)+1), columns=['rank'])
        output_df = pd.concat([output_df, important_features], axis=1)
        output_df.to_csv(folder_path + 'features_importance.csv', index=False, header=True)