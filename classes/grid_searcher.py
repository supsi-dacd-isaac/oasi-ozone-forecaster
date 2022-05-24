import os

import numpy as np
import pandas as pd

from multiprocessing import Queue, Process

queue_results = Queue()


def gs_cell_process(mt, q, features, region, target_column, df_x, df_y, weights):
    lcl_kpis, lcl_prediction = mt.training_cross_validated_fs(features, region, target_column, df_x, df_y, weights)

    # Write on the queue
    q.put(
            {
                'lcl_kpis': lcl_kpis,
                'lcl_prediction': lcl_prediction
            }
        )

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

    def search_weights(self, region, target_column, cfg_file_name):
        """
        Iterate over the weights as specified in the config file, and for each iteration save the KPIs and all the
        prediction of the algorithm performed on the test set
        """

        self.get_datasets()

        for key, df in self.dataFrames.items():

            fp = self.input_gatherer.output_folder_creator(key)
            fn = fp + 'GS_KPIs_' + target_column + '_' + cfg_file_name.split(os.sep)[-1].replace('.json', '') + '.csv'
            fn_pred = fp + 'GS_all_errs_' + target_column + '_' + cfg_file_name.split(os.sep)[-1].replace('.json', '') + '.csv'

            # Initialize empty files in folder. Data will be inserted at each iteration step
            if self.cfg['regions'][region]['gridSearcher']['hyperParsOptimizationNGB'] is None:
                pd.DataFrame([], columns=['w1', 'w2', 'w3', 'Accuracy_1', 'Accuracy_2',
                                          'Accuracy_3', 'Accuracy', 'RMSE1', 'RMSE2',
                                          'RMSE3', 'RMSE', 'MAE1', 'MAE2', 'MAE3', 'MAE','ConfMat']).to_csv(fn, mode='a', header=True, index=False)
                # pd.DataFrame([], columns=['w1', 'w2', 'w3', 'Fold', 'Measurements', 'Prediction']).to_csv(fn_pred, mode='w',
                #                                                                                           header=True, index=False)
            else:
                pd.DataFrame([], columns=['w1', 'w2', 'w3', 'ne', 'le', 'Accuracy_1', 'Accuracy_2',
                                          'Accuracy_3', 'Accuracy', 'RMSE1', 'RMSE2',
                                          'RMSE3', 'RMSE', 'MAE1', 'MAE2', 'MAE3', 'MAE','ConfMat']).to_csv(fn, mode='a', header=True, index=False)
                # pd.DataFrame([], columns=['w1', 'w2', 'w3', 'ne', 'le', 'Fold', 'Measurements', 'Prediction']).to_csv(fn_pred,
                #                                                                                                       mode='w',
                #                                                                                                       header=True,
                #                                                                                                       index=False)

            l1 = np.arange(self.cfg['regions'][region]['gridSearcher']['w1_start'],
                           self.cfg['regions'][region]['gridSearcher']['w1_end']+1,
                           self.cfg['regions'][region]['gridSearcher']['w1_step'])
            l2 = np.arange(self.cfg['regions'][region]['gridSearcher']['w2_start'],
                           self.cfg['regions'][region]['gridSearcher']['w2_end']+1,
                           self.cfg['regions'][region]['gridSearcher']['w2_step'])
            l3 = np.arange(self.cfg['regions'][region]['gridSearcher']['w3_start'],
                           self.cfg['regions'][region]['gridSearcher']['w3_end']+1,
                           self.cfg['regions'][region]['gridSearcher']['w3_step'])

            _, _, features, df_x, df_y = self.features_analyzer.dataset_splitter(key, df, target_column)

            # Run the grid search
            procs = []
            for w1 in l1:
                for w2 in l2:
                    for w3 in l3:
                        self.logger.info('Region: %s, target: %s -> weights = [%i, %i, %i]' % (region, target_column,
                                                                                             w1, w2, w3))
                        weights = {'w1': w1, 'w2': w2, 'w3': w3}

                        tmp_proc = Process(target=gs_cell_process, args=[self.model_trainer, queue_results,
                                                                         features, region, target_column, df_x, df_y,
                                                                         weights])
                        # tmp_proc.start()
                        procs.append(tmp_proc)

            self.logger.info('Start the processes (n=%i)' % len(procs))
            for proc in procs:
                proc.start()

            self.logger.info('Join the processes together')
            for proc in procs:
                proc.join()

            # Read from the queue
            i = 0
            results = []
            while True:
                item = queue_results.get()
                results.append(item)
                i += 1
                self.logger.warning('Read data from the queue, added item n. %i' % i)
                if i == len(procs):
                    break

            self.logger.warning('Save the results on file %s' % fn)
            for result in results:
                result['lcl_kpis'].to_csv(fn, mode='a', header=False, index=False, quoting=2)

