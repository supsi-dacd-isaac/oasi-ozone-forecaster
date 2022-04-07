import os

import numpy as np
import pandas as pd

class GridSearcher:
    """
    Given a dataset with a target column this class performs grid search over the weights w1, w2 and w3 as specified in
    the config files
    """

    def __init__(self, features_analyzer, input_gatherer, model_trainer, forecast_type, cfg, logger):
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
        self.input_gatherer = input_gatherer
        self.model_trainer = model_trainer
        self.forecast_type = forecast_type
        self.cfg = cfg
        self.logger = logger
        self.dataFrames = None

    def get_datasets(self):
        self.dataFrames = self.features_analyzer.dataFrames

    def search_weights(self, region, target_column):
        """
        Iterate over the weights as specified in the config file, and for each iteration save the KPIs and all the
        prediction of the algorithm performed on the test set
        """

        self.get_datasets()

        for key, df in self.dataFrames.items():

            fp = self.input_gatherer.output_folder_creator(key)
            fn = fp + fp.split(os.sep)[1] + '_grid_search_KPIs_' + self.cfg['regions'][region]['gridSearcher']['typeGridSearch'] + '.csv'
            fn_pred = fp + fp.split(os.sep)[1] + '_grid_search_all_errors_' + self.cfg['regions'][region]['gridSearcher']['typeGridSearch'] + '.csv'

            # Initialize empty files in folder. Data will be inserted at each iteration step
            pd.DataFrame([],
                         columns=['w1', 'w2', 'w3', 'Accuracy_1', 'Accuracy_2', 'Accuracy_3', 'Accuracy', 'RMSE',
                                  'MAE', 'ConfMat']).to_csv(fn, mode='w', header=True, index=False)
            pd.DataFrame([],
                         columns=['w1', 'w2', 'w3', 'Fold', 'Measurements', 'Prediction']).to_csv(fn_pred, mode='w', header=True, index=False)

            l1 = np.arange(self.cfg['regions'][region]['gridSearcher']['w1_start'], self.cfg['regions'][region]['gridSearcher']['w1_end']+1, self.cfg['regions'][region]['gridSearcher']['w1_step'])
            l2 = np.arange(self.cfg['regions'][region]['gridSearcher']['w2_start'], self.cfg['regions'][region]['gridSearcher']['w2_end']+1, self.cfg['regions'][region]['gridSearcher']['w2_step'])
            l3 = np.arange(self.cfg['regions'][region]['gridSearcher']['w3_start'], self.cfg['regions'][region]['gridSearcher']['w3_end']+1, self.cfg['regions'][region]['gridSearcher']['w3_step'])

            _, _, features, df_x, df_y = self.features_analyzer.dataset_splitter(key, df, target_column)

            for w1 in l1:
                for w2 in l2:
                    for w3 in l3:
                        self.cfg['regions'][region]['featuresAnalyzer']['w1'] = w1
                        self.cfg['regions'][region]['featuresAnalyzer']['w2'] = w2
                        self.cfg['regions'][region]['featuresAnalyzer']['w3'] = w3
                        self.logger.info('Weights=[%i, %i, %i]' % (w1, w2, w3))

                        if self.cfg['regions'][region]['gridSearcher']['typeGridSearch'] == 'multiple':
                            lcl_KPIs, lcl_prediction = self.model_trainer.training_cross_validated_multiple_FS(features, region, df_x, df_y)
                        elif self.cfg['regions'][region]['gridSearcher']['typeGridSearch'] == 'single':
                            lcl_KPIs, lcl_prediction = self.model_trainer.training_cross_validated_single_FS(features, region, df_x, df_y)
                        elif self.cfg['regions'][region]['gridSearcher']['typeGridSearch'] == 'test':
                            lcl_KPIs, lcl_prediction = self.model_trainer.get_KPIs_with_random_separation(features, region, df_x, df_y)
                        else:
                            self.logger.error(
                                'Option for grid search type is not valid. Available options are "multiple", "single" or "test"')
                            return

                        lcl_KPIs.to_csv(fn, mode='a', header=False, index=False, quoting=2)
                        lcl_prediction.to_csv(fn_pred, mode='a', header=False, index=False)
