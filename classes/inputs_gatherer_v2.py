# import section
import os
import json
import glob
from datetime import datetime, timedelta
from time import sleep

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

import constants


class InputsGathererV2:
    """
    Class handling the gathering of the inputs needed by a collection of predictors.
    There are 3 ways to create a dataframe:
    
    - Read an existing CSV (see method dataframe_reader)
    - Define a region composed of measurements and forecast stations, define the signals to be used by each station,
      then create all possible signals in JSON format and finally create the dataframe by querying InfluxDB (see method dataframe_builder_regions)
    - read an existing JSON containing a set of signals and create the dataframe by querying InfluxDB (see method dataframe_builder_custom)
    """

    def __init__(self, influxdb_client, cfg, logger, artificial_features):
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

        self.cfg = cfg
        self.logger = logger
        self.io_data = None
        self.io_data_sub = None
        self.predictor_input = None
        self.day_to_predict = None
        self.cfg_signals = None
        self.artificial_features = artificial_features
        self.input_sequence = None

        # Constants
        self.OK = 0
        self.MISSING_DATA = 1
        self.FORECAST_SURROGATION = 2

    # def build_global_input_dataset(self):
    #     """
    #     Build the dataset
    #     """
    #     self.io_data = []
    #     self.io_data_sub = []
    #     self.day_to_predict = None
    #
    #     # Create the signals list considering all the couples location-model
    #     self.cfg_signals = dict(signals=[])
    #
    #     # Cycle over the folders
    #     for tmp_folder in glob.glob('%s%s*%s' % (self.cfg['folders']['models'], os.sep, self.forecast_type)):
    #
    #         # Check if the current folder refers to a location configured for the prediction
    #         region_code = tmp_folder.split(os.sep)[-1].split('_')[0]
    #         if region_code in self.cfg['regions'].keys():
    #
    #             # Cycle over the input files in the folder (each files correspond to a model)
    #             for input_cfg_file in glob.glob('%s%s/inputs_*.json' % (tmp_folder, os.sep)):
    #                 tmp_cfg_signals = json.loads(open(input_cfg_file).read())
    #                 self.cfg_signals['signals'] = self.cfg_signals['signals'] + tmp_cfg_signals['signals']
    #
    #     self.cfg_signals['signals'] = list(set(self.cfg_signals['signals']))
    #
    #     # get the values in the DB
    #     i = 1
    #     for signal in self.cfg_signals['signals']:
    #         self.logger.info('Try to add input n. %04d/%04d, %s' % (i, len(self.cfg_signals['signals']), signal))
    #         self.add_input_value(signal=signal, forecast_substitution=False)
    #         self.logger.info('Added input n. %04d/%04d' % (i, len(self.cfg_signals['signals'])))
    #         i += 1
    #
    #     # Check the data availability
    #     self.check_inputs_availability()

    def build_dataset_for_prediction(self, region, input_signals, day_to_predict):
        end_day = datetime.strptime(day_to_predict, '%Y-%m-%d')
        # Go back for two days to be safe, maybe the following parameters should be configurable
        start_day = end_day - timedelta(days=3)
        curr_day = start_day
        self.predictor_input = None
        while True:
            self.io_data = []
            for i in range(0, len(input_signals)):
                self.add_input_value(signal=input_signals[i], day=curr_day.strftime('%Y-%m-%d'))
                self.logger.info('Added input n. %04d/%04d' % (i + 1, len(input_signals)))

            merged_io_data = pd.concat(self.io_data, axis=1)
            merged_io_data.index.name = 'time'

            if self.predictor_input is None:
                self.predictor_input = merged_io_data
            else:
                self.predictor_input = pd.concat([self.predictor_input, merged_io_data], axis=0)

            curr_day += timedelta(days=1)

            # Last day-1d checking
            if curr_day > end_day:
                break

        # Remove eventual duplicated columns
        self.predictor_input = self.predictor_input.loc[:, ~self.predictor_input.columns.duplicated()]


    def build_dataset(self, name, input_signals, day_to_predict=False):
        """
        Build the training dataset given a signal json file in folder "conf/dataset" either from a region or from a
        custom list
        """

        fp = self.output_folder_creator(name)
        file_name_df = fp + fp.split(os.sep)[1] + '_dataset.csv'

        if os.path.isfile(file_name_df):
            os.unlink(file_name_df)

        # initialize the Pandas dataframe that will contain the final dataset
        output_signals = self.cfg['regions'][name]['targetColumns']

        dataset = pd.DataFrame(columns=['date'] + input_signals + output_signals)

        # Iterate over the years
        for year in self.cfg['datasetSettings']['years']:
            start_day = str(year) + '-' + self.cfg['datasetSettings']['startDay']
            end_day = str(year) + '-' + self.cfg['datasetSettings']['endDay']
            start_dt = datetime.strptime(start_day, '%Y-%m-%d')
            end_dt = datetime.strptime(end_day, '%Y-%m-%d')

            # Check if you have to jump to the next year
            if start_dt > end_dt:
                end_day = str(year+1) + '-' + self.cfg['datasetSettings']['endDay']
                end_dt = datetime.strptime(end_day, '%Y-%m-%d')

            # todo this code should be checked carefully
            # In 2020 we lost part of the forecasted signal, so we're forced to discard most days until the 17th August
            if len(output_signals) > 0:
                if 'O3' in output_signals[0] and year == 2020 and start_dt < datetime.strptime('2020-08-17', '%Y-%m-%d'):
                    start_day = '2020-08-17'
                if 'O3' in output_signals[0] and year == 2020 and end_dt < datetime.strptime('2020-08-17', '%Y-%m-%d'):
                    continue

            curr_day = start_day
            while True:
                self.io_data = []

                # Iterate over the input signals
                for i in range(0, len(input_signals)):
                    self.add_input_value(signal=input_signals[i], day=curr_day, forecast_substitution=True)
                    self.logger.info('Added input n. %04d/%04d' % (i+1, len(input_signals)))
                    sleep(self.cfg['datasetSettings']['sleepTimeBetweenQueries'])

                merged_io_data = pd.concat(self.io_data, axis=1)
                merged_io_data.index.name = 'time'

                if os.path.isfile(file_name_df) is False:
                    merged_io_data.to_csv(file_name_df, mode='w', header=True, index=True)
                else:
                    merged_io_data.to_csv(file_name_df, mode='a', header=False, index=True)

                # Add a day
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
            #  Check if the value is None or nan
            if self.io_data[k] is None or np.isnan(self.io_data[k]):
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

        if self.cfg['predictionGeneralSettings']['meanImputation']['case'] == 'fixed':
            query = 'SELECT mean(value) FROM %s WHERE code=\'%s\' AND case=\'%s\' AND ' \
                    'time>=\'%s\' AND time<\'%s\'' % (self.cfg['influxDB']['measurementInputsHistory'], code,
                                                      self.forecast_type,
                                                      self.cfg['predictionGeneralSettings']['meanImputation']['since'],
                                                      datetime.fromtimestamp(self.day_to_predict).strftime('%Y-%m-%d'))
        else:
            days_to_go_back = int(self.cfg['predictionGeneralSettings']['meanImputation']['case'].replace('last', '')[:-1])
            since = datetime.fromtimestamp(self.day_to_predict) - timedelta(days_to_go_back)
            query = 'SELECT mean(value) FROM %s WHERE code=\'%s\' AND case=\'%s\' AND ' \
                    'time>=\'%s\' AND time<\'%s\'' % (self.cfg['influxDB']['measurementInputsHistory'], code,
                                                      self.forecast_type, since.strftime('%Y-%m-%d'),
                                                      datetime.fromtimestamp(self.day_to_predict).strftime('%Y-%m-%d'))
        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')

        try:
            return float(res.raw['series'][0]['values'][0][1])
        except Exception as e:
            self.logger.error('Impossible to calculate the mean of the past values')
            return np.nan

    def add_input_value(self, signal, day, forecast_substitution=False, force_substitution=False):
        """
        Add the value related to a given input signal

        :param signal: signal code
        :type signal: string
        :return: query
        :rtype: string
        """

        # Measurements
        signal_case = signal.split('__')[-1]
        if signal_case == 'meas':
            self.do_measurement_query(signal, signal, day)
        # MeteoSuisse predictions
        elif signal_case == 'ms-pred':
            self.do_forecast_query(signal, day)
        # Copernicus predictions
        elif signal_case == 'cop-pred':
            self.logger.warning('Copernicus data handling currently not available (\"%s\")' % signal)

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


    def do_forecast_period_query(self, signal_data, measurement, forecast_substitution, force_substitution=False):
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
        self.calc_data(query=query, signal_data=signal_data, func=func, forecast_substitution=forecast_substitution,
                       str_steps=str_steps, str_dt=str_dt, force_substitution=force_substitution)

    @staticmethod
    def create_forecast_chunk_steps_string(chunk, case):

        start = constants.CHUNKS_FORECASTS[case][chunk]['start']
        end = constants.CHUNKS_FORECASTS[case][chunk]['end']
        str_steps = '('
        for i in range(start, end + 1):
            str_steps += 'step=\'step%02d\' OR ' % i

        str_steps = '%s)' % str_steps[:-4]
        return str_steps

    def do_copernicus_step_query(self, signal_data, measurement):
        (location, signal_code, step) = signal_data.split('__')

        dt = self.set_forecast_day()

        dt = dt.replace(minute=0, hour=0, second=0, microsecond=0)
        step = 'step%02d' % int(step[4:])

        # MOR case: for the prediction of a generic day D (performed at D_07:30) the latest COPERNICUS forecast
        # is the one available at ~00:30 AM of D, related to D-1,D,D+1,D+2,...
        if self.forecast_type == 'MOR':
            dt = dt - timedelta(days=1)
        # EVE case: for the prediction of a generic day D (performed at D-1_19:30) the latest COPERNICUS forecast
        # is the one available at ~00:30 AM of D-1, related to D-2,D-1,D,D+1,...
        elif self.forecast_type == 'EVE':
            dt = dt - timedelta(days=2)

        str_dt = dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND step=\'%s\' AND ' \
                'time=\'%s\'' % (measurement, location, signal_code, step, str_dt)

        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')
        if 'series' in res.raw.keys() and len(res.raw['series']) > 0:
            try:
                val = float(res.raw['series'][0]['values'][0][1])
                self.io_data[signal_data] = val
            except Exception as e:
                self.logger.error('Data not available')
                self.io_data[signal_data] = np.nan
        else:
            self.logger.error('Data not available')
            self.io_data[signal_data] = np.nan

    def do_forecast_step_query(self, signal_data, measurement, forecast_substitution, force_substitution=False):
        if len(signal_data.split('__')) == 4:
            (location, signal_code, case, step) = signal_data.split('__')
        else:
            (location, signal_code, step) = signal_data.split('__')

        dt = self.set_forecast_day()

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            # COSMO2 forecast start at 00:00, COSMO1 at 03:00
            if '_c2' in signal_code:
                dt = dt.replace(minute=0, hour=0, second=0, microsecond=0)
                step = 'step%03d' % int(step[4:])
                time_interval = 3
            else:
                dt = dt.replace(minute=0, hour=3, second=0, microsecond=0)
                step = 'step%02d' % int(step[4:])
                time_interval = 1
        else:
            dt = dt.replace(minute=0, hour=12, second=0, microsecond=0)
            if '_c2' in signal_code:
                step = 'step%03d' % int(step[4:])
                time_interval = 3
            else:
                step = 'step%02d' % int(step[4:])
                time_interval = 1

        str_dt = dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        if force_substitution is False:
            query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND step=\'%s\' AND ' \
                    'time=\'%s\'' % (measurement, location, signal_code, step, str_dt)

            self.logger.info('Performing query: %s' % query)
            res = self.influxdb_client.query(query, epoch='s')

            if 'series' in res.raw.keys() and len(res.raw['series']) > 0:
                try:
                    # The training of Meteosuisse temperatures were done in Kelvin degrees
                    # todo this thing just below is a shame, the data in the DB to use for the training are in Celsius!!
                    # if signal_code in ['TD_2M', 'T_2M']:
                    #     val = float(res.raw['series'][0]['values'][0][1]) + 273.1
                    # else:
                    #     val = float(res.raw['series'][0]['values'][0][1])
                    val = float(res.raw['series'][0]['values'][0][1])
                    self.io_data[signal_data] = val
                except Exception as e:
                    self.logger.error('Data not available')
                    self.io_data[signal_data] = np.nan
            else:
                self.logger.warning('Forecast not available')
                if forecast_substitution is True:
                    # Manage the substitution
                    self.logger.warning('The forecast will be substituted by a correspondent measured value')
                    dt_start_meas = dt + timedelta(hours=int(step.replace('step', '')))
                    dt_end_meas = dt_start_meas + timedelta(hours=time_interval)

                    res_meas = self.do_forecast_substitution(location, signal_code, dt_start_meas, dt_end_meas, None)
                    try:
                        val = float(res_meas.raw['series'][0]['values'][0][1])
                        self.io_data[signal_data] = val
                    except Exception as e:
                        self.logger.error('Data not available')
                        self.io_data[signal_data] = np.nan
                else:
                    self.logger.warning('No data from query %s' % query)
                    self.io_data[signal_data] = np.nan
        else:
            self.logger.warning('FORCED SUBSTITUTION: The forecast will be substituted by a correspondent measured value')
            dt_start_meas = dt + timedelta(hours=int(step.replace('step', '')))
            dt_end_meas = dt_start_meas + timedelta(hours=time_interval)
            res_meas = self.do_forecast_substitution(location, signal_code, dt_start_meas, dt_end_meas, None)
            if res_meas is not None and 'series' in res_meas.raw.keys():
                try:
                    val = float(res_meas.raw['series'][0]['values'][0][1])
                    self.io_data_sub[signal_data] = val
                except Exception as e:
                    self.logger.error('EXCEPTION: %s' % str(e))
                    self.logger.error('Unable to find a substitute for signal %s' % signal_data)
                    self.io_data_sub[signal_data] = np.nan
            else:
                self.io_data_sub[signal_data] = np.nan

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
            if self.forecast_type == 'EVE':
                day_date = dt + timedelta(int(day[-1])+1)
            else:
                day_date = dt + timedelta(int(day[-1]))
        else:
            # INPUT:  D-x <-- D
            if self.forecast_type == 'EVE':
                day_date = dt - timedelta(int(day[-1])-1)
            else:
                day_date = dt - timedelta(int(day[-1]))

        query = 'SELECT mean(value) FROM %s WHERE location=\'%s\' AND ' \
                'signal=\'%s\' AND time>=\'%s\' AND time<=\'%s\' ' \
                'GROUP BY time(1h)' % (measurement, location, signal_code,
                                       '%sT00:00:00Z' % day_date.strftime('%Y-%m-%d'),
                                       '%sT23:59:59Z' % day_date.strftime('%Y-%m-%d'))
        self.calc_data(query=query, signal_data=signal_data, func=func, forecast_substitution=False, str_steps=None,
                       str_dt=None)

    @staticmethod
    def set_MOR_EVE_daytime(forecast_type, dt):
        if forecast_type == 'MOR':
            # Data starting is assigned to 04 UTC
            return dt.replace(hour=4)
        elif forecast_type == 'EVE':
            # Data starting is assigned to 16 UTC
            return dt.replace(hour=16)

    def do_moving_average_query(self, signal_data, measurement):
        (location, signal_code, moving_average_func, hours) = signal_data.split('__')

        end_dt = self.set_forecast_day()
        end_dt = self.set_MOR_EVE_daytime(self.forecast_type, end_dt)
        end_dt = end_dt - timedelta(hours=int(hours[1:]))

        start_dt = end_dt - timedelta(hours=int(int(moving_average_func.replace('moving_average', ''))))

        query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time>=\'%s\' AND ' \
                'time<=\'%s\'' % (measurement, location, signal_code,
                                  '%s:30:00Z' % start_dt.strftime('%Y-%m-%dT%H'),
                                  '%s:00:00Z' % end_dt.strftime('%Y-%m-%dT%H'))
        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')
        try:
            # Consider all the available data
            vals = []
            for elem in res.raw['series'][0]['values']:
                vals.append(elem[1])
            self.io_data[signal_data] = np.mean(vals)
        except Exception as e:
            self.logger.error('No data from query %s' % query)
            self.io_data[signal_data] = np.nan

    def do_forecast_query(self, signal_data, day):
        (location, signal_code, suffix) = signal_data.split('__')
        measurement = self.cfg['influxDB']['measurementInputsForecasts']

        start_time = '%sT00:00:00Z' % day
        end_time = '%sT23:59:59Z' % day
        query = ('SELECT value, step FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND '
                 'time>=\'%s\' AND time<=\'%s\'') % (measurement, location, signal_code, start_time, end_time)
        self.logger.info('Performing query: %s' % query)
        day_before = (datetime.strptime(day, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        query_day_before = (('SELECT value, step FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time=\'%s\'') %
                            (measurement, location, signal_code, '%sT12:00:00Z' % day_before))
        self.logger.info('Performing query: %s' % query_day_before)
        try:
            res = self.influxdb_client.query(query, epoch='us')
            res_db = self.influxdb_client.query(query_day_before, epoch='us')
            df = res[measurement]
            df_db = res_db[measurement]
            # Delete not meaningful values if needed
            df = df[df['value'] != -99999.0]
            df_db = df_db[df_db['value'] != -99999.0]

            # Handling of the second run of the day before, we need the predictions of 00:00 (step12),
            # 01:00 (step13) and 02:00 (step14) (also step15 for GLOB signal)
            # The irradiance (GLOB) forecast of MeteoSuisse does not provide the step0 value! I agree with you,
            # it's a shame.
            if signal_code == 'GLOB':
                df_db = df_db[df_db['step'].isin(['step12', 'step13', 'step14', 'step15'])].copy()
                df_db['step_number'] = df_db['step'].str.extract('(\d+)').astype(int)
                df_db.index = pd.Timestamp('%s 12:00:00+00:00' % day_before) + pd.to_timedelta(df_db['step_number'], unit='h')
                df = df[df['step'] != 'step00']
            else:
                df_db = df_db[df_db['step'].isin(['step12', 'step13', 'step14'])].copy()
                df_db['step_number'] = df_db['step'].str.extract('(\d+)').astype(int)
                df_db.index = pd.Timestamp('%s 12:00:00+00:00' % day_before) + pd.to_timedelta(df_db['step_number'], unit='h')
            df_db['step_number'] -= 15

            # Handling of daily first run (03:00)
            df_first_run = df.loc['%s 03:00:00+00:00' % day].copy()
            df_first_run['step_number'] = df_first_run['step'].str.extract('(\d+)').astype(int)
            df_first_run.index = pd.Timestamp('%s 03:00:00+00:00' % day) + pd.to_timedelta(df_first_run['step_number'], unit='h')
            # We consider only the hours of the present day
            df_first_run = pd.concat([df_db, df_first_run.loc[df_first_run['step_number'] < 21]])

            # Handling of daily second run (12:00)
            df_second_run = df.loc['%s 12:00:00+00:00' % day].copy()
            df_second_run['step_number'] = df_second_run['step'].str.extract('(\d+)').astype(int)
            df_second_run.index = pd.Timestamp('%s 12:00:00+00:00' % day) + pd.to_timedelta(df_second_run['step_number'], unit='h')
            # We consider only the hours of the present day. Besides, for the first 12 hours of the day the second run
            # predictions are not available, being performed at 12:00, so the first run ones are used.
            if signal_code == 'GLOB':
                df_second_run = pd.concat([df_first_run.loc[df_first_run['step_number'] < 10], df_second_run.loc[df_second_run['step_number'] < 12]])
            else:
                df_second_run = pd.concat([df_first_run.loc[df_first_run['step_number'] < 9], df_second_run.loc[df_second_run['step_number'] < 12]])

            df_first_run = df_first_run.drop(columns=['step', 'step_number'])
            df_first_run = df_first_run.rename(columns={'value': '%s_1run' % signal_data})
            df_first_run['%s_1run__flag' % signal_data] = df_first_run['%s_1run' % signal_data].isna().astype(int)

            df_second_run = df_second_run.drop(columns=['step', 'step_number'])
            df_second_run = df_second_run.rename(columns={'value': '%s_2run' % signal_data})
            df_second_run['%s_2run__flag' % signal_data] = df_second_run['%s_2run' % signal_data].isna().astype(int)

            self.io_data.append(df_first_run)
            self.io_data.append(df_second_run)

        except Exception as e:
            self.logger.error('Exception: %s' % str(e))
            self.logger.error('Forecast not available for %s' % signal_data)
            substitute_signal_data = '%s__%s__meas' % (self.cfg['forecastedSubstitutes']['stations'][location],
                                                       self.cfg['forecastedSubstitutes']['signals'][signal_code]['id'])

            self.logger.error('Try to surrogate the forecast with its substitute %s' % substitute_signal_data)
            self.do_measurement_query(substitute_signal_data, '%s_1run' % signal_data, day)
            self.do_measurement_query(substitute_signal_data, '%s_2run' % signal_data, day)

    def do_measurement_query(self, signal_data_query, signal_data_output_df, day):
        (location, signal_code, suffix) = signal_data_query.split('__')
        measurement = self.cfg['influxDB']['measurementInputsMeasurements']

        start_time = '%sT00:00:00Z' % day
        end_time = '%sT23:59:59Z' % day
        query = ('SELECT mean(value) AS value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND '
                 'time>=\'%s\' AND time<=\'%s\' GROUP BY time(1h)') % (measurement, location, signal_code, start_time,
                                                                       end_time)
        self.logger.info('Performing query: %s' % query)
        try:
            res = self.influxdb_client.query(query, epoch='us')
            df = res[measurement]
            df.index.name = 'time'
            df = df.rename(columns={'value': signal_data_output_df})

            # Check if we are trying to surrogate a forecast
            if signal_data_query == signal_data_output_df:
                df['%s__flag' % signal_data_output_df] = df[signal_data_output_df].isna().astype(int)
            else:
                df['%s__flag' % signal_data_output_df] = self.FORECAST_SURROGATION

            self.io_data.append(df)
        except Exception as e:
            self.logger.error('Exception: %s' % str(e))
            date_range = pd.date_range(start=start_time, end=end_time, freq='H', tz='UTC')
            data = {
                signal_data_output_df: [np.nan] * len(date_range),
                '%s__flag' % signal_data_output_df: [self.MISSING_DATA] * len(date_range)
            }
            self.io_data.append(pd.DataFrame(data, index=date_range))

    def generate_input_signals_codes(self, region):
        """
        Method to generate and save all known signals of a specific region (e.g. Ticino) with defined measuring and
        forecasting stations
        """
        signal_list = []
        # Add measures signals codes
        for meas_station in self.cfg["regions"][region]["measureStations"]:
            for signal in self.cfg["measuredSignalsStations"][meas_station].keys():
                signal_list.append('%s__%s__meas' % (meas_station, signal))

        for pred_station in self.cfg["regions"][region]["forecastStations"]:
            for signal in self.cfg["forecastedSignalsStations"][pred_station]:
                signal_list.append('%s__%s__ms-pred' % (pred_station, signal))

        # Add Copernicus forecast signals codes
        for cop_station in self.cfg["regions"][region]["copernicusStations"]:
            for cop_signal in self.cfg["copernicusSignalsStations"][cop_station]:
                signal_list.append('%s__%s__cop-pred' % (cop_station, cop_signal))

        return signal_list

    def output_folder_creator(self, dataset_name):
        """
        Get the address of the output folder for the current case
        """
        # Check if start day > end date (it means they are related to different years)
        start_dt = datetime.strptime('1970-%s' % self.cfg['datasetSettings']['startDay'], '%Y-%m-%d')
        end_dt = datetime.strptime('1970-%s' % self.cfg['datasetSettings']['endDay'], '%Y-%m-%d')
        if start_dt > end_dt:
            inc_year = 1
        else:
            inc_year = 0

        folder_path = '%s%s_%s-%s_%s-%s%s' % (self.cfg['outputFolder'], dataset_name,
                                              self.cfg['datasetSettings']['years'][0],
                                              self.cfg['datasetSettings']['startDay'],
                                              self.cfg['datasetSettings']['years'][-1]+inc_year,
                                              self.cfg['datasetSettings']['endDay'], os.sep)
        folder_path = folder_path.replace('-', '')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return folder_path

    def dataframe_builder_regions(self, day_to_predict=False):
        for region in self.cfg['regions']:
            input_signals = self.generate_input_signals_codes(region)

            if day_to_predict is False:
                self.build_dataset(name=region, input_signals=input_signals, day_to_predict=day_to_predict)
            else:
                self.build_dataset_for_prediction(region=region, input_signals=input_signals,
                                                  day_to_predict=day_to_predict)


    def retrieve_signals_from_files(self, predictor_code):
        meas_cfg = {}
        pred_cfg = {}
        self.input_sequence = {}

        for sig_file in glob.glob('%s/*%s*___signals.json' % (self.cfg['folders']['models'], predictor_code)):
            data = json.loads(open(sig_file).read())
            predictor_code = sig_file.split(os.sep)[1].replace('__out___signals.json', '')
            self.input_sequence[predictor_code] = data['input']

            for input_sig in data['input']:
                tmp = input_sig.split('__')
                if tmp[2] == 'meas':
                    if tmp[0] not in meas_cfg.keys():
                        meas_cfg[tmp[0]] = {}
                    if tmp[1] not in meas_cfg:
                        meas_cfg[tmp[0]][tmp[1]] = ''

                elif tmp[2] == 'ms-pred':
                    if tmp[0] not in pred_cfg.keys():
                        pred_cfg[tmp[0]] = []
                    if tmp[1] not in pred_cfg:
                        pred_cfg[tmp[0]].append(tmp[1])

        self.cfg['regions'] =  {
            'ALL': {
                'measureStations': list(meas_cfg.keys()),
                'forecastStations': list(pred_cfg.keys()),
                'copernicusStations': []
            }
        }
        self.cfg['measuredSignalsStations'] = meas_cfg
        self.cfg['forecastedSignalsStations'] = pred_cfg
