# import section
import json
import scipy.io as sio
import numpy as np
import pytz
import os


from influxdb import InfluxDBClient
from datetime import date, datetime, timedelta

import constants

class DatasetBuilder:
    """
    Builder of the input dataset for the forecaster
    """

    def __init__(self, influxdb_client, cfg, logger, forecast_type, predictors_folder):
        """
        Constructor

        :param influxdb_client: InfluxDB client
        :type influxdb_client: InfluxDBClient
        :param cfg: FTP parameters for the files exchange
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        :param forecast_type: forecast type (MOR | EVE)
        :type forecast_type: string
        :param predictors_folder: folder where the predictors are located with the related inputs
        :type predictors_folder: string
        """
        # set the variables
        self.influxdb_client = influxdb_client
        self.cfg = cfg
        self.forecast_type = forecast_type
        self.logger = logger
        self.input_data = dict()
        self.input_data_desc = []
        self.input_data_values = []
        self.day_to_predict = None
        self.cfg_signals = None
        self.predictors_folder = predictors_folder

    def build(self, location):
        """
        Build the dataset for a given station

        :param location: location to predict
        :type location: string
        """
        self.logger.info('Dataset creation for location %s' % location)
        self.input_data = dict()
        self.input_data_desc = []
        self.input_data_values = []
        self.day_to_predict = None

        # load the signals to use in the prediction for the proper couple location_case (e.g. CHI_MOR)
        sigs_file = '%s/json/%s_%s.json' % (self.predictors_folder, location['code'], self.forecast_type)

        self.cfg_signals = json.loads(open(sigs_file).read())
        location['signals'] = self.cfg_signals['signals']

        # get the values in the DB
        for signal in location['signals']:
            self.add_input_value(signal=signal)

        # organize the array
        self.input_data_desc = location['signals']
        self.input_data_values = []
        for signal in location['signals']:
            self.input_data_values.append(self.input_data[signal])

        # Check the surrogate_on features of the predictors
        tmp = self.predictors_folder.split(os.sep)
        predictor = tmp[-1]

        # check if the  feature is set in the configuration file
        if 'predictorsFeatures' in location.keys() and \
            predictor in location['predictorsFeatures'][self.forecast_type].keys() and \
            'surrogate' in location['predictorsFeatures'][self.forecast_type][predictor]:

            # check if the surrogate feature is off
            if location['predictorsFeatures'][self.forecast_type][predictor]['surrogate'] == 'off':
                # if surrogate is not enabled then do not perform the prediction
                if None in self.input_data_values:
                    self.input_data = dict()
                    self.input_data_desc = []
                    self.input_data_values = []
        else:
            # To be sure consider this case as a surrogate off
            if None in self.input_data_values:
                self.input_data = dict()
                self.input_data_desc = []
                self.input_data_values = []

    def add_input_value(self, signal):
        """
        Add the input value related to a given signal

        :param signal: signal code
        :type signal: string
        :return query
        :rtype string
        """
        # Signals exception (e.g. isWeekend, etc.)
        if signal in constants.SIGNAL_EXCEPTIONS:
            self.handle_exception_signal(signal)
        else:
            tmp = signal.split('__')

            # Forecasts data
            if tmp[0] in constants.METEO_FORECAST_STATIONS:
                measurement = self.cfg['influxDB']['measurementMeteoSuisse']

                if '__step' in signal:
                    self.do_forecast_step_query(signal, measurement)

                else:
                    self.do_forecast_period_query(signal, measurement)

            # Measurement data
            else:
                measurement = self.cfg['influxDB']['measurementOASI']

                if '__d0' in signal or '__d1' in signal or '__d2' in signal or \
                   '__d3' in signal or '__d4' in signal or '__d5' in signal:
                    # check if there are chunks
                    if '__chunk' in signal:
                        self.do_chunk_query(signal, measurement)
                    else:
                        self.do_daily_query(signal, measurement)
                else:
                    self.do_hourly_query(signal, measurement)

    def do_forecast_period_query(self, signal_data, measurement):
        (location, signal_code, chunk, func) = signal_data.split('__')

        dt = self.set_forecast_day()

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            str_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
            str_steps = self.create_forecast_chunk_steps_string(chunk, '03')

        else:
            str_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')
            str_steps = self.create_forecast_chunk_steps_string(chunk, '12')

        query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND ' \
                'time=\'%s\'' % (measurement, location, signal_code, str_steps, str_dt)
        self.calc_data(query=query, signal_data=signal_data, func=func)

    @staticmethod
    def create_forecast_chunk_steps_string(chunk, case):

        start = constants.CHUNKS_FORECASTS[case][chunk]['start']
        end = constants.CHUNKS_FORECASTS[case][chunk]['end']
        str_steps = '('
        for i in range(start, end+1):
            str_steps += 'step=\'step%02d\' OR ' % i

        str_steps = '%s)' % str_steps[:-4]
        return str_steps

    def do_forecast_step_query(self, signal_data, measurement):
        # todo to change when wordings like 'TICIA__CLCT__chunk2__step10' will not be used anymore
        if len(signal_data.split('__')) == 4:
            (location, signal_code, case, step) = signal_data.split('__')
        else:
            (location, signal_code, step) = signal_data.split('__')
        # wirie step parameter with the proper format
        step = 'step%02d' % int(step[4:])

        dt = self.set_forecast_day()

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            str_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
        else:
            str_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')

        if signal_code == 'MTR':
            pass

        query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND step=\'%s\' AND ' \
                'time=\'%s\'' % (measurement, location, signal_code, step, str_dt)

        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')

        try:
            # The training of Meteosuisse temperatures were done in Kelvin degrees
            if signal_code in ['TD_2M', 'T_2M']:
                val = float(res.raw['series'][0]['values'][0][1]) + 273.1
            else:
                val = float(res.raw['series'][0]['values'][0][1])
            self.input_data[signal_data] = val
        except Exception as e:
            self.logger.error('Forecast not available')
            self.logger.error('No data from query %s' % query)
            self.input_data[signal_data] = np.nan

    def do_chunk_query(self, signal_data, measurement):
        (location, signal_code, day, chunk, func) = signal_data.split('__')

        # Chunk management
        dt = self.set_forecast_day()
        dt = dt.date()

        if chunk == 'chunk1':
            start_dt = dt - timedelta(int(day[-1]))
            end_dt = dt - timedelta(int(day[-1])-1)
            start_tm = '23:00:00'
            end_tm = '04:59:00'

        elif chunk == 'chunk2':
            start_dt = dt - timedelta(int(day[-1]))
            end_dt = dt - timedelta(int(day[-1]))
            start_tm = '17:00:00'
            end_tm = '22:59:00'

        elif chunk == 'chunk3':
            start_dt = dt - timedelta(int(day[-1]))
            end_dt = dt - timedelta(int(day[-1]))
            start_tm = '11:00:00'
            end_tm = '16:59:00'

        elif chunk == 'chunk4':
            start_dt = dt - timedelta(int(day[-1]))
            end_dt = dt - timedelta(int(day[-1]))
            start_tm = '05:00:00'
            end_tm = '10:59:00'

        query = 'SELECT mean(value) FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND ' \
                'time>=\'%s\' AND time<=\'%s\' ' \
                'GROUP BY time(1h)' % (measurement, location, signal_code,
                                       '%sT%sZ' % (start_dt.strftime('%Y-%m-%d'), start_tm),
                                       '%sT%sZ' % (end_dt.strftime('%Y-%m-%d'), end_tm))

        self.calc_data(query=query, signal_data=signal_data, func=func)

    def do_daily_query(self, signal_data, measurement):
        if len(signal_data.split('__')) == 4:
            (location, signal_code, day, func) = signal_data.split('__')
        else:
            # cases of past YO3 and YO3_index
            (location, signal_code, day) = signal_data.split('__')
            func = 'mean'

        dt = self.set_forecast_day()
        day_date = dt - timedelta(int(day[-1]))

        query = 'SELECT mean(value) FROM %s WHERE location=\'%s\' AND ' \
                'signal=\'%s\' AND time>=\'%s\' AND time<=\'%s\' ' \
                'GROUP BY time(1h)' % (measurement, location, signal_code,
                                       '%sT00:00:00Z' % day_date.strftime('%Y-%m-%d'),
                                       '%sT23:59:59Z' % day_date.strftime('%Y-%m-%d'))
        self.calc_data(query=query, signal_data=signal_data, func=func)

    def do_hourly_query(self, signal_data, measurement):
        (location, signal_code, hours) = signal_data.split('__')

        dt = self.set_forecast_day()

        if self.forecast_type == 'MOR':
            # Data starting is assigned to 04 UTC
            dt = dt.replace(hour=4)
        elif self.forecast_type == 'EVE':
            # Data starting is assigned to 16 UTC
            dt = dt.replace(hour=16)

        if hours[0] == 'm':
            dt = dt - timedelta(hours=int(hours[1:]))
        else:
            dt = dt + timedelta(hours=int(hours[1:]))

        query = 'SELECT mean(value) FROM %s WHERE location=\'%s\' AND ' \
                'signal=\'%s\' AND time>=\'%s\' AND time<=\'%s\'' % (measurement, location, signal_code,
                                                                     '%s:00:00Z' % dt.strftime('%Y-%m-%dT%H'),
                                                                     '%s:59:59Z' % dt.strftime('%Y-%m-%dT%H'))
        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')

        try:
            self.input_data[signal_data] = res.raw['series'][0]['values'][0][1]
        except Exception as e:
            self.logger.error('Forecast not available')
            self.logger.error('No data from query %s' % query)
            self.input_data[signal_data] = np.nan

    def save(self, output_file):
        """
        Save the input datasets in a MAT file

        :param output_file: output file
        :type output_file: string
        """
        try:
            # check if all the data are available
            if len(self.input_data_values) > 0:
                sio.savemat(output_file, {'desc': self.input_data_desc, 'values': self.input_data_values})
        except Exception as e:
            self.logger.error('Unable to save data in the file %s' % output_file)

    def calc_data(self, query, signal_data, func):
        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')

        vals = []
        tmp = signal_data.split('__')
        try:
            for i in range(0, len(res.raw['series'][0]['values'])):
                if res.raw['series'][0]['values'][i][1] is not None:

                    # The training of Meteosuisse temperatures were done in Kelvin degrees
                    if tmp[1] in ['TD_2M', 'T_2M']:
                        val = float(res.raw['series'][0]['values'][i][1]) + 273.1
                    else:
                        val = float(res.raw['series'][0]['values'][i][1])
                    vals.append(val)

            if func == 'min':
                self.input_data[signal_data] = np.min(vals)
            elif func == 'max':
                self.input_data[signal_data] = np.max(vals)
            elif func == 'mean':
                self.input_data[signal_data] = np.mean(vals)
        except Exception as e:
            self.logger.error('Forecast not available')
            self.logger.error('No data from query %s' % query)
            self.input_data[signal_data] = np.nan

    def handle_exception_signal(self, signal_data):

        dt = self.set_forecast_day()

        # RHW
        if 'RHW' in signal_data:
            (signal_code, day) = signal_data.split('__')
            day_date = dt - timedelta(int(day[-1]))
            str_date = day_date.strftime('%Y-%m-%d')

            measurement = self.cfg['influxDB']['measurementGlobal']
            query = 'SELECT value FROM %s WHERE signal=\'%s\' AND time=\'%sT00:00:00Z\'' % (measurement, signal_code,
                                                                                            str_date)

            self.logger.info('Performing query: %s' % query)
            res = self.influxdb_client.query(query, epoch='s')

            try:
                self.input_data[signal_data] = res.raw['series'][0]['values'][0][1]
            except Exception as e:
                self.logger.error('Forecast not available')
                self.logger.error('No data from query %s' % query)
                self.input_data[signal_data] = np.nan

        else:
            # if EVE case the day to predict is the next one
            if self.forecast_type == 'EVE':
                dt = dt + timedelta(1)

            # day of the week
            if signal_data == 'DayWeek':

                self.input_data[signal_data] = float(dt.weekday())

            # weekend/not weekend
            elif signal_data == 'IsWeekend':

                if dt.weekday() >= 5:
                    # weekend day
                    self.input_data[signal_data] = 1.0
                else:
                    # not weekend day
                    self.input_data[signal_data] = 0.0

            # holydays
            elif signal_data == 'IsHolyday':

                if dt.strftime('%m-%d') in constants.HOLYDAYS:
                    self.input_data[signal_data] = 1.0
                else:
                    self.input_data[signal_data] = 0.0

    def save_training_data(self, location):
        # check if new data are available
        if len(self.input_data_values) > 0:
            str_new_vals = '%s' % datetime.strftime(datetime.now(), '%Y-%m-%d')

            for val in self.input_data_values:
                str_new_vals = '%s,%f' % (str_new_vals, val)

            # create a new the temporary file where the inputs will be saved
            inputs_tmp_file = '%s/tmp/%s_%s.csv' % (self.cfg['local']['trainingDatasets'], location['code'],
                                                    self.forecast_type)
            self.logger.info('Create input temporary file %s' % inputs_tmp_file)
            with open(inputs_tmp_file, 'w') as fw:
                fw.write('%s\n' % str_new_vals)
            fw.close()

    def set_forecast_day(self):
        """
        Set the day related to the forecast
        """

        dt = datetime.now(pytz.utc)
        if self.cfg['forecastPeriod']['case'] != 'current':
            # Customize the prediction date
            (y, m, d) = self.cfg['dayToForecast'].split('-')
            dt = dt.replace(year=int(y), month=int(m), day=int(d))

        # Format the global variable as a string
        if self.day_to_predict is None:
            # morning case
            if self.forecast_type == 'MOR':
                self.day_to_predict = dt

            # evening case
            else:
                self.day_to_predict = dt + timedelta(days=1)

            self.day_to_predict = self.day_to_predict.replace(hour=0, minute=0, second=0)
            self.day_to_predict = int(self.day_to_predict.timestamp())
        return dt
