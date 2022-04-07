# import section
import glob
import json
import os
from datetime import datetime, timedelta
from time import sleep

import numpy as np
import pandas as pd
import pytz
from influxdb import InfluxDBClient

import constants


class InputsGatherer:
    """
    Class handling the gathering of the inputs needed by a collection of predictors.
    There are 3 ways to create a dataframe:
    
    - Read an existing CSV (see method dataframe_reader)
    - Define a region composed of measurements and forecast stations, define the signals to be used by each station, then create all possible signals in JSON format and finally create the dataframe by querying InfluxDB (see method dataframe_builder_regions)
    - read an existing JSON containing a set of signals and create the dataframe by querying InfluxDB (see method dataframe_builder_custom)
    """

    def __init__(self, influxdb_client, forecast_type, cfg, logger, artificial_features):
        """
        Constructor

        :param influxdb_client: InfluxDB client
        :type influxdb_client: InfluxDBClient
        :param forecast_type: Forecast type (MOR | EVE)
        :type forecast_type: str
        :param cfg: FTP parameters for the files exchange
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        :param artificial_features: Artificial features
        :type artificial_features: ArtificialFeatures
        """
        # set the variables
        self.influxdb_client = influxdb_client
        self.forecast_type = forecast_type
        self.cfg = cfg
        self.logger = logger
        self.io_data = None
        self.day_to_predict = None
        self.cfg_signals = None
        self.artificial_features = artificial_features

    def build_global_input_dataset(self):
        """
        Build the dataset
        """
        self.io_data = dict()
        self.day_to_predict = None

        # Get the locations configured for the prediction
        locations_configured = []
        for location in self.cfg['locations']:
            locations_configured.append(location['code'])

        # Create the signals list considering all the couples location-model
        self.cfg_signals = dict(signals=[])

        # Cycle over the folders
        for tmp_folder in glob.glob('%s%s*%s' % (self.cfg['folders']['models'], os.sep, self.forecast_type)):

            # Check if the current folder refers to a location configured for the prediction
            location_code = tmp_folder.split(os.sep)[-1].split('_')[0]
            if location_code in locations_configured:

                # Cycle over the input files in the folder (each files correspond to a model)
                for input_cfg_file in glob.glob('%s%s/inputs_*.json' % (tmp_folder, os.sep)):
                    tmp_cfg_signals = json.loads(open(input_cfg_file).read())
                    self.cfg_signals['signals'] = self.cfg_signals['signals'] + tmp_cfg_signals['signals']

        self.cfg_signals['signals'] = list(set(self.cfg_signals['signals']))

        # get the values in the DB
        i = 1
        for signal in self.cfg_signals['signals']:
            self.logger.info('Try to add input n. %02d/0%2d, %s' % (i, len(self.cfg_signals['signals']), signal))
            self.add_input_value(signal=signal)
            self.logger.info('Added input n. %02d/0%2d' % (i, len(self.cfg_signals['signals'])))
            i += 1

        # Check the data availability
        self.check_inputs_availability()

    def build_dataset(self, name, input_signals):
        """
        Build the training dataset given a signal json file in folder "conf/dataset" either from a region or from a
        custom list
        """

        self.io_data = dict()
        fp = self.output_folder_creator(name)
        file_name_df = fp + fp.split(os.sep)[1] + '_dataset.csv'
        file_name_tc = fp + fp.split(os.sep)[1] + '_target_columns.csv'

        # Set the signal list
        # self.input_signals = dict()
        # self.cfg_signals['signals'] = signal_list

        # # destroy repetitions but preserve the order
        # input_signals = list(dict.fromkeys(self.cfg_signals['signals']))

        # initialize the Pandas dataframe that will contain the final dataset
        output_signals = self.cfg['regions'][name]['targetColumns']

        dataset = pd.DataFrame(columns=['date'] + input_signals + output_signals)

        # Boolean flag to determine whether we should override an existing output dataset csv or not
        flag_starting_dataset = True

        # Iterate over the years
        for year in self.cfg['datasetSettings']['years']:
            start_day = str(year) + '-' + self.cfg['datasetSettings']['startDay']
            end_day = str(year) + '-' + self.cfg['datasetSettings']['endDay']
            start_dt = datetime.strptime(start_day, '%Y-%m-%d')
            end_dt = datetime.strptime(end_day, '%Y-%m-%d')

            # In 2020 we lost part of the forecasted signal, so we're forced to discard most days until the 17th August
            if year == 2020 and start_dt < datetime.strptime('2020-08-17', '%Y-%m-%d'):
                start_day = '2020-08-17'

            if year == 2020 and end_dt < datetime.strptime('2020-08-17', '%Y-%m-%d'):
                continue

            curr_day = start_day

            while True:

                self.cfg['dayToForecast'] = curr_day

                # Iterate over the input signals
                for i in range(0, len(input_signals)):
                    self.add_input_value(signal=input_signals[i])
                    self.logger.info('Added input n. %04d/%04d' % (i+1, len(input_signals)))
                    sleep(self.cfg['datasetSettings']['sleepTimeBetweenQueries'])

                # Iterate over the output signals
                for i in range(0, len(output_signals)):
                    self.add_output_value(signal=output_signals[i])
                    self.logger.info('Added output n. %04d/%04d' % (i+1, len(output_signals)))
                    sleep(self.cfg['datasetSettings']['sleepTimeBetweenQueries'])

                lcl_data = dict({'date': curr_day}, **self.io_data)
                lcl_df = pd.DataFrame([lcl_data], columns=dataset.columns   )

                if flag_starting_dataset:
                    lcl_df.to_csv(file_name_df, mode='w', header=True, index=False)
                    flag_starting_dataset = False
                else:
                    lcl_df.to_csv(file_name_df, mode='a', header=False, index=False)

                dataset = dataset.append(lcl_df)

                # add a day
                curr_dt = datetime.strptime(curr_day, '%Y-%m-%d')
                curr_day = datetime.strftime(curr_dt + timedelta(days=1), '%Y-%m-%d')

                # Last day-1d checking
                if curr_dt.timestamp() >= end_dt.timestamp():
                    break

        return dataset

    def read_dataset(self, csv_file):

        dataset = pd.read_csv(csv_file, sep=',')
        return dataset

    def check_inputs_availability(self):
        dps = []

        self.io_data_availability = dict()
        for k in self.io_data.keys():
            if np.isnan(self.io_data[k]):
                self.io_data[k] = self.retrieve_past_mean(code=k)
                self.io_data_availability[k] = False
            else:
                self.io_data_availability[k] = True
                point = {
                    'time': int(self.day_to_predict),
                    'measurement': self.cfg['influxDB']['measurementInputsHistory'],
                    'fields': dict(value=float(self.io_data[k])),
                    'tags': dict(code=k, case=self.forecast_type)
                }
                dps.append(point)

        if len(dps) > 0:
            self.logger.info('Inserted in the history %i inputs values related predictions of day %s, '
                             'case %s' % (len(dps), datetime.fromtimestamp(self.day_to_predict).strftime('%Y-%m-%d'),
                                          self.forecast_type))
            self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    def retrieve_past_mean(self, code):

        query = 'SELECT mean(value) FROM %s WHERE code=\'%s\' AND case=\'%s\' AND ' \
                'time>=\'%s\' AND time<\'%s\'' % (self.cfg['influxDB']['measurementInputsHistory'], code,
                                                  self.forecast_type,
                                                  self.cfg['predictionSettings']['startDateForMeanImputation'],
                                                  datetime.fromtimestamp(self.day_to_predict).strftime('%Y-%m-%d'))

        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')

        try:
            return float(res.raw['series'][0]['values'][0][1])
        except Exception as e:
            self.logger.error('Impossible to calculate the mean of the past values')
            return np.nan

    def add_input_value(self, signal):
        """
        Add the value related to a given input signal

        :param signal: signal code
        :type signal: string
        :return: query
        :rtype: string
        """

        # Signals exception (e.g. isWeekend, etc.)
        if signal in constants.SIGNAL_EXCEPTIONS or any([s in signal for s in constants.ARTIFICIAL_FEATURES]):
            self.handle_exception_signal(signal)
        else:
            tmp = signal.split('__')

            # Forecasts data
            if tmp[0] in constants.METEO_FORECAST_STATIONS:
                # measurement = self.cfg['influxDB']['measurementMeteoSuisse']
                measurement = self.cfg['influxDB']['measurementInputsForecasts']

                if '__step' in signal:
                    self.do_forecast_step_query(signal, measurement)

                else:
                    self.do_forecast_period_query(signal, measurement)

            # Measurement data
            else:
                # if tmp[0][0:2] == 'MS':
                #     measurement = self.cfg['influxDB']['measurementMeteoSuisse']
                # else:
                #     measurement = self.cfg['influxDB']['measurementOASI']
                measurement = self.cfg['influxDB']['measurementInputsMeasurements']

                if '__d0' in signal or '__d1' in signal or '__d2' in signal or \
                        '__d3' in signal or '__d4' in signal or '__d5' in signal:
                    # check if there are chunks
                    # check if there are chunks
                    if '__chunk' in signal:
                        self.do_chunk_query(signal, measurement)
                    else:
                        self.do_daily_query(signal, measurement)
                else:
                    # specific period query
                    if 'h__' in signal:
                        self.do_period_query(signal, measurement, 'measure')
                    else:
                        self.do_hourly_query(signal, measurement)

    def add_output_value(self, signal):
        """
        Add the value related to a given output signal

        :param signal: signal code
        :type signal: string
        :return: query
        :rtype: string
        """
        measurement = self.cfg['influxDB']['measurementInputsMeasurements']
        self.do_daily_query(signal, measurement, flag_output_signal=True)


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
        for i in range(start, end + 1):
            str_steps += 'step=\'step%02d\' OR ' % i

        str_steps = '%s)' % str_steps[:-4]
        return str_steps

    def do_forecast_step_query(self, signal_data, measurement):
        # todo to change when wordings like 'TICIA__CLCT__chunk2__step10' will not be used anymore
        if len(signal_data.split('__')) == 4:
            (location, signal_code, case, step) = signal_data.split('__')
        else:
            (location, signal_code, step) = signal_data.split('__')
        # write step parameter with the proper format
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
            self.io_data[signal_data] = val
        except Exception as e:
            self.logger.error('Forecast not available')
            self.logger.error('No data from query %s' % query)
            self.io_data[signal_data] = np.nan

    def do_chunk_query(self, signal_data, measurement):
        (location, signal_code, day, chunk, func) = signal_data.split('__')

        # Chunk management
        dt = self.set_forecast_day()
        dt = dt.date()

        if chunk == 'chunk1':
            start_dt = dt - timedelta(int(day[-1]))
            end_dt = dt - timedelta(int(day[-1]) - 1)
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

    def do_daily_query(self, signal_data, measurement, flag_output_signal=False):
        if len(signal_data.split('__')) == 4:
            (location, signal_code, day, func) = signal_data.split('__')
        else:
            # cases of past YO3 and YO3_index
            (location, signal_code, day) = signal_data.split('__')
            func = 'mean'

        dt = self.set_forecast_day()

        # If I have an output signal I have to get data related to future days
        if flag_output_signal is True:
            # OUTPUT: D --> D+x
            day_date = dt + timedelta(int(day[-1]))
        else:
            # INPUT:  D-x <-- D
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
            self.io_data[signal_data] = res.raw['series'][0]['values'][0][1]
        except Exception as e:
            self.logger.error('Forecast not available')
            self.logger.error('No data from query %s' % query)
            # Nocturnal irradiance in BIO is missing rather than 0, so fixing it here (DM 09.11.21)
            if location == 'BIO' and signal_code == 'Gl':
                self.io_data[signal_data] = 0.0
            else:
                self.io_data[signal_data] = np.nan

    def do_period_query(self, signal_data, measurement, case):
        (location, signal_code, period, func) = signal_data.split('__')

        dt = self.set_forecast_day()

        if self.forecast_type == 'MOR':
            # Data starting is assigned to 04 UTC
            dt = dt.replace(hour=4)
        elif self.forecast_type == 'EVE':
            # Data starting is assigned to 16 UTC
            dt = dt.replace(hour=16)

        hours_num = int(period[0:-1])

        if case == 'measure':
            dt_start = dt - timedelta(hours=hours_num)
            dt_end = dt
        else:
            dt_start = dt
            dt_end = dt + timedelta(hours=hours_num)

        query = 'SELECT %s(value) FROM %s WHERE location=\'%s\' AND ' \
                'signal=\'%s\' AND time>=\'%s\' AND time<=\'%s\'' % (func, measurement, location, signal_code,
                                                                     '%s:00:00Z' % dt_start.strftime('%Y-%m-%dT%H'),
                                                                     '%s:59:59Z' % dt_end.strftime('%Y-%m-%dT%H'))
        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')

        try:
            self.io_data[signal_data] = res.raw['series'][0]['values'][0][1]
        except Exception as e:
            self.logger.error('Forecast not available')
            self.logger.error('No data from query %s' % query)
            self.io_data[signal_data] = np.nan

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
                self.io_data[signal_data] = np.min(vals)
            elif func == 'max':
                self.io_data[signal_data] = np.max(vals)
            elif func == 'mean':
                self.io_data[signal_data] = np.mean(vals)
        except Exception as e:
            self.logger.error('Forecast not available')
            self.logger.error('No data from query %s' % query)
            self.io_data[signal_data] = np.nan

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
                self.io_data[signal_data] = res.raw['series'][0]['values'][0][1]
            except Exception as e:
                self.logger.error('Forecast not available')
                self.logger.error('No data from query %s' % query)
                self.io_data[signal_data] = np.nan

        elif any([s in signal_data for s in constants.ARTIFICIAL_FEATURES]):
            res = self.artificial_features.analyze_signal(signal_data)
            self.io_data[signal_data] = res

        else:
            # if EVE case the day to predict is the next one
            if self.forecast_type == 'EVE':
                dt = dt + timedelta(1)

            # day of the week
            if signal_data == 'DayWeek':

                self.io_data[signal_data] = float(dt.weekday())

            # weekend/not weekend
            elif signal_data == 'IsWeekend':

                if dt.weekday() >= 5:
                    # weekend day
                    self.io_data[signal_data] = 1.0
                else:
                    # not weekend day
                    self.io_data[signal_data] = 0.0

            # holydays
            elif signal_data == 'IsHolyday':

                if dt.strftime('%m-%d') in constants.HOLYDAYS:
                    self.io_data[signal_data] = 1.0
                else:
                    self.io_data[signal_data] = 0.0

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

    def get_previous_day(self):
        if self.cfg['forecastPeriod']['case'] == 'current':
            dt = datetime.now(pytz.utc) - timedelta(1)
        else:
            dt = datetime.strptime(self.cfg['dayToForecast'], '%Y-%m-%d')
            dt = pytz.utc.localize(dt)
            dt = dt - timedelta(1)
        return dt

    def calc_yesterday_o3_daily_values(self):
        """
        Calc daily data of O3 values related to yesterday
        """
        # get the date to analyze
        dt = self.get_previous_day()

        query = 'SELECT mean(value) FROM %s WHERE signal=\'O3\' AND time>=\'%sT00:00:00Z\' AND ' \
                'time<=\'%sT23:59:59Z\' GROUP BY time(1h), location, signal' % (self.cfg['influxDB']['measurementOASI'],
                                                                                dt.strftime('%Y-%m-%d'),
                                                                                dt.strftime('%Y-%m-%d'))

        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')

        utc_ts = int(dt.timestamp())
        dps = []
        for series in res.raw['series']:
            vals = []
            for i in range(0, len(series['values'])):
                if series['values'][i][1] is not None:
                    vals.append(series['values'][i][1])
            daily_max = np.max(vals)
            daily_idx = self.get_index(np.max(vals))

            point = {
                'time': utc_ts,
                'measurement': self.cfg['influxDB']['measurementOASI'],
                'fields': dict(value=float(daily_max)),
                'tags': dict(signal='YO3', location=series['tags']['location'])
            }
            dps.append(point)

            point = {
                'time': utc_ts,
                'measurement': self.cfg['influxDB']['measurementOASI'],
                'fields': dict(value=float(daily_idx)),
                'tags': dict(signal='YO3_index', location=series['tags']['location'])
            }
            dps.append(point)

        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    @staticmethod
    def get_index(val):
        """
        Get ozone index value (http://www.oasi.ti.ch/web/dati/aria.html)

        :param val: ozone value
        :type val: float
        :return: ozone index
        :rtype: int
        """
        if 0 <= val <= 60:
            return 1
        elif 60 < val <= 120:
            return 2
        elif 120 < val <= 135:
            return 3
        elif 135 < val <= 180:
            return 4
        elif 180 < val <= 240:
            return 5
        else:
            return 6

    def hourly_measured_signals(self, measurementStation, measuredSignal):
        signals = []
        for i in range(24):
            signals.append(measurementStation + '__' + measuredSignal + '__m' + str(i))
        return signals

    def hourly_forecasted_signals(self, forecastStation, forecastedSignal):
        signals = []
        for i in range(34):
            signals.append(forecastStation + '__' + forecastedSignal + '__step' + str(i))
        return signals

    def chunks_forecasted_signals(self, forecastStation, forecastedSignal):
        signals = []
        for i in range(1, 5):
            for modifier in ['min', 'max', 'mean']:
                signals.append(forecastStation + '__' + forecastedSignal + '__chunk' + str(i) + '__' + modifier)
        return signals

    # def artificial_features_measured_signals(self, measurementStation):
    #     signals = []
    #     if measurementStation != 'MS-LUG':
    #         signals.append(measurementStation + '__NOx__12h_mean')
    #         signals.append(measurementStation + '__NO2__24h_mean')
    #         if measurementStation != 'LUG':
    #             signals.append(measurementStation + '__YO3__d1')
    #             signals.append(measurementStation + '__YO3_index__d1')
    #     return signals

    def past_days_means_measured_signals(self, measurementStation, measuredSignal):
        signals = []
        for i in ['24h', '48h', '72h']:
            signals.append(measurementStation + '__' + measuredSignal + '__' + str(i) + '__mean')
        return signals

    def artificial_features_forecasted_signals(self, forecastStation):
        signals = []
        signals.append(forecastStation + '__CLCT__mean_mor')
        signals.append(forecastStation + '__CLCT__mean_eve')
        signals.append(forecastStation + '__GLOB__mean_mor')
        signals.append(forecastStation + '__GLOB__mean_eve')
        signals.append(forecastStation + '__TOT_PREC__sum')
        signals.append(forecastStation + '__T_2M__12h_mean')
        signals.append(forecastStation + '__T_2M__12h_mean_squared')
        signals.append(forecastStation + '__T_2M__MAX')
        signals.append(forecastStation + '__TD_2M__MAX')
        signals.append(forecastStation + '__TD_2M__transf')

        return signals

    def generate_input_signals_codes(self, region):
        """
        Method to generate and save all known signals of a specific region (e.g. Ticino) with defined measuring and
        forecasting stations
        """

        signal_list = []

        for measurementStation in self.cfg["regions"][region]["MeasureStations"]:
            # signal_list.extend(self.artificial_features_measured_signals(measurementStation))
            for measuredSignal in self.cfg["measuredSignalsStations"][measurementStation]:
                signal_list.extend(self.past_days_means_measured_signals(measurementStation, measuredSignal))
                signal_list.extend(self.hourly_measured_signals(measurementStation, measuredSignal))

        for forecastStation in self.cfg["regions"][region]["ForecastStations"]:
            signal_list.extend(self.artificial_features_forecasted_signals(forecastStation))
            for forecastedSignal in self.cfg["forecastedSignalsStations"][forecastStation]:
                signal_list.extend(self.hourly_forecasted_signals(forecastStation, forecastedSignal))
                signal_list.extend(self.chunks_forecasted_signals(forecastStation, forecastedSignal))
        signal_list.extend(self.cfg['globalSignals'])

        return signal_list

    def output_folder_creator(self, dataset_name):
        """
        Get the address of the output folder for the current case
        """

        folder_path = '%s%s_%s_%s-%s_%s-%s%s' % (self.cfg['outputFolder'],
                                                 dataset_name,
                                                 self.forecast_type,
                                                 self.cfg['datasetSettings']['years'][0],
                                                 self.cfg['datasetSettings']['startDay'],
                                                 self.cfg['datasetSettings']['years'][-1],
                                                 self.cfg['datasetSettings']['endDay'], os.sep)
        folder_path = folder_path.replace('-', '')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return folder_path

    def dataframe_builder_regions(self):

        for region in self.cfg['regions']:
            input_signals = self.generate_input_signals_codes(region)
            self.build_dataset(name=region, input_signals=input_signals)

    def dataframe_builder_custom(self):

        for dataset in self.cfg['datasetSettings']['customJSONSignals']:
            name = dataset['filename'].split('.')[0]
            fn = self.cfg['datasetSettings']["loadSignalsFolder"] + dataset['filename']
            self.build_dataset(name=name, signals_file=fn)
