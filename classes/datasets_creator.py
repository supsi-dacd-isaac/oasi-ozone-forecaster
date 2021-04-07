# import section
import csv
import pandas as pd

from datetime import datetime, timedelta
from influxdb import InfluxDBClient

from classes.dataset_builder import DatasetBuilder

class DatasetsCreator:
    """
    Create a set of dataset for a given period
    """

    def __init__(self, cfg, logger):
        """
        Constructor

        :param cfg: FTP parameters for the files exchange
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        """
        # set the variables
        self.cfg = cfg
        self.logger = logger

        # --------------------------------------------------------------------------- #
        # InfluxDB connection
        # --------------------------------------------------------------------------- #
        self.logger.info('Connection to InfluxDb server on socket [%s:%s]' % (cfg['influxDB']['host'], cfg['influxDB']['port']))
        self.influx_client = InfluxDBClient(host=self.cfg['influxDB']['host'],
                                            port=self.cfg['influxDB']['port'],
                                            password=self.cfg['influxDB']['password'],
                                            username=self.cfg['influxDB']['user'],
                                            database=self.cfg['influxDB']['database'])
        self.logger.info('Connection successful')

    def create(self, start_day, end_day):
        """
        Create the datasets files

        :param start_day: start day (%Y-%m-%d)
        :type start_day: string
        :param end_day: end day (%Y-%m-%d)
        :type end_day: string
        """
        forecast_types = self.cfg['cases']
        dt_start = datetime.strptime(start_day, '%Y-%m-%d')
        dt_end = datetime.strptime(end_day, '%Y-%m-%d')
        for location in self.cfg['locations']:
            for forecast_type in forecast_types:

                dt_curr = dt_start

                self.logger.info('Create dataset for couple [%s:%s]' % (location['code'], forecast_type))

                header = None

                # define the output file
                output_file = '%s/%s_%s.csv' % (self.cfg['local']['outputFolder'], location['code'], forecast_type)
                flag_header = False

                if self.cfg['local']['operationOnOutputFile'] == 'w':
                    days_already_saved = []
                else:
                    days_already_saved = self.get_days_already_saved(output_file)

                while dt_curr != dt_end:

                    # check the month
                    if 5 <= dt_curr.month <= 9:
                        self.cfg['dayToForecast'] = dt_curr.strftime('%Y-%m-%d')

                        self.logger.info('Day %s' % self.cfg['dayToForecast'])

                        # check if the day is not already saved
                        if self.cfg['dayToForecast'] not in days_already_saved:

                            dsb = DatasetBuilder(influxdb_client=self.influx_client, cfg=self.cfg, logger=self.logger,
                                                 forecast_type=forecast_type,
                                                 predictors_folder=self.cfg['local']['signalsConfigFolder'])
                            dsb.build(location=location)

                            # save the header
                            if header is None:
                                signals_code = dsb.cfg_signals['signals']
                                signals_code.insert(0, 'date')
                                header = signals_code

                            # check if a new file has to be created
                            if self.cfg['local']['operationOnOutputFile'] == 'w':
                                # write the file headers
                                if flag_header is False:
                                    with open(output_file, mode='w') as csv_fw:
                                        csv_writer = csv.writer(csv_fw, delimiter=',')
                                        csv_writer.writerow(header)
                                        flag_header = True
                                    csv_fw.close()

                            dsb.input_data_values.insert(0, self.cfg['dayToForecast'])

                            vals = []
                            for code_desc in header:
                                if code_desc != 'date':
                                    if code_desc in dsb.input_data.keys():
                                        vals.append(dsb.input_data[code_desc])
                                else:
                                    vals.append(dsb.input_data_values[0])

                            # check if all the data are available
                            if len(header) == len(vals):

                                # write the data row
                                with open(output_file, mode='a') as csv_fw:
                                    csv_writer = csv.writer(csv_fw, delimiter=',')
                                    csv_writer.writerow(vals)
                                csv_fw.close()
                            else:
                                self.logger.error('Dataset not available for day %s' % self.cfg['dayToForecast'])

                            del dsb

                        else:
                            self.logger.error('Day %s already saved in dataset file' % self.cfg['dayToForecast'])
                    else:
                        # you should be at the beginning of October, go to next 5th of May
                        dt_curr = dt_curr.replace(year=int(dt_curr.year)+1, month=5, day=5)

                    # go to the next day
                    dt_curr = dt_curr + timedelta(days=1)

    @staticmethod
    def get_days_already_saved(data_file):
        """
        Return a list with the days already saved in the file

        :param data_file: path of the data file
        :type data_file: string
        :return: array containing the already saved day
        :rtype: list
        """
        days = []

        with open(data_file, mode='r') as csv_fr:
            csv_reader = csv.reader(csv_fr, delimiter=',')
            for row in csv_reader:
                days.append(row[0])
        csv_fr.close()

        return days
