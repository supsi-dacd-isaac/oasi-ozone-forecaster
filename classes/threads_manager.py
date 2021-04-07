# import section
import threading
import subprocess
import multiprocessing
import math
import os

from glob import glob

def forecast(launcher, mr_folder, predictor_folder, predictor_code, predictor_name, inbox, outbox):
    """
    Launch the forecasts script
    """
    subprocess.run([launcher, mr_folder, predictor_folder, predictor_code, predictor_name, inbox, outbox])

class ThreadsManager:
    """
    Manager of the threads that launch the Matlab forecasters
    """

    def __init__(self, cfg, logger, forecast_type):
        """
        Constructor

        :param cfg: configuration parameters
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        :param forecast_type: forecast type (MOR | EVE)
        :type forecast_type: string
        """
        # set the variables
        self.cfg = cfg
        self.logger = logger
        self.forecast_type = forecast_type

    def run(self):
        """
        Run the threads
        """
        ths = []
        self.logger.info('Launch the threads for all the locations')
        for location in self.cfg['locations']:
            # check if the input dataset is ready for the location (i.e. all the input values are available)

            # cycle over all the predictors codes (latest + the pasts)
            for predictors_folder in glob('%s/*' % self.cfg['matlab']['forecastersFolder']):

                tmp = predictors_folder.split('/')

                input_file = '%s/%s_%s_%s.mat' % (self.cfg['local']['inputMat'], location['code'],
                                                  self.forecast_type, tmp[-1])
                if os.path.isfile(input_file):

                    self.logger.info('Perform forecast for location %s, predictor %s, case %s' % (location['code'],
                                                                                                  tmp[-1],
                                                                                                  self.forecast_type))

                    ths.append(threading.Thread(target=forecast, args=(self.cfg['matlab']['scriptLauncher'],
                                                                       self.cfg['matlab']['runtimeEnvironmentFolder'],
                                                                       self.cfg['matlab']['forecastersFolder'],
                                                                       tmp[-1],
                                                                       '%s_%s' % (location['code'], self.forecast_type),
                                                                       self.cfg['local']['inputMat'],
                                                                       self.cfg['local']['outputMat'])))
                else:
                    self.logger.error('Data not available for location %s, '
                                      'predictor %s, case %s' % (location['code'], tmp[-1], self.forecast_type))
                    self.logger.error('Skip the forecast')

        # full threads batches
        num_cpus = multiprocessing.cpu_count()
        full_batches = math.floor(len(ths) / num_cpus)
        i = 0
        for i in range(1, full_batches+1):
            # launch a full batch of threads
            self.run_threads_batch(ths=ths, start=(num_cpus*i)-num_cpus, end=num_cpus*i)

        # bath with the remaining threads
        self.run_threads_batch(ths=ths, start=num_cpus*i, end=len(ths))

        self.logger.info('All the threads ended')

    def run_threads_batch(self, start, end, ths):
        self.logger.info('Launch in parallel threads [%i:%i]' % (start, end))

        # Set the threads batch
        ths_batch = ths[start:end]

        # Start the threads
        self.logger.info('Start the threads')
        for th in ths_batch:
            th.start()

        # Join the threads
        self.logger.info('Join the threads')
        for th in ths_batch:
            th.join()
