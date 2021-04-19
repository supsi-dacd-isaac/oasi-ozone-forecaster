# import section
import os
import datetime
import tailhead
import copy
import pytz
import time
import ftplib
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from datetime import date, datetime, timedelta
from influxdb import InfluxDBClient
from sklearn.utils import check_array

import constants

class DataManager:
    """
    Generic interface for data (measures and forecasts) management
    """

    def __init__(self, influxdb_client, cfg, logger):
        """
        Constructor

        :param influxdb_client: InfluxDB client
        :type influxdb_client: InfluxDBClient
        :param cfg: FTP parameters for the files exchange
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        """
        # set the variables
        self.influxdb_client = influxdb_client
        self.cfg = cfg
        self.logger = logger

        # Define the time zone
        self.tz_local = pytz.timezone(self.cfg['local']['timeZone'])

    def get_raw_files(self):

        if os.path.isdir(self.cfg['ftp']['localFolders']['tmp']) is False:
            os.mkdir(self.cfg['ftp']['localFolders']['tmp'])

        try:
            # perform FTP connection and login
            ftp = ftplib.FTP(self.cfg['ftp']['host'])
            ftp.login(self.cfg['ftp']['user'], self.cfg['ftp']['password'])

            for ftp_dir in [self.cfg['ftp']['remoteFolders']['measures'],
                            self.cfg['ftp']['remoteFolders']['forecasts']]:
                self.logger.info('Getting files via FTP from %s/%s' % (self.cfg['ftp']['host'], ftp_dir))
                ftp.cwd('/%s' % ftp_dir)

                # cycle over the remote files
                for file_name in ftp.nlst('*'):
                    tmp_local_file = os.path.join(self.cfg['ftp']['localFolders']['tmp'] + "/", file_name)
                    try:
                        self.logger.info('%s -> %s/%s/%s' % (file_name, os.getcwd(),
                                                             self.cfg['ftp']['localFolders']['tmp'], file_name))

                        # get the file from the server
                        with open(tmp_local_file, 'wb') as f:
                            def callback(data):
                                f.write(data)

                            ftp.retrbinary("RETR " + file_name, callback)

                        # delete the remote file
                        # ftp.delete(file_name)
                    except Exception as e:
                        self.logger.error('Downloading exception: %s' % str(e))
        except Exception as e:
            self.logger.error('Connection exception: %s' % str(e))

        # close the FTP connection
        ftp.close()

    def insert_data(self):
        self.logger.info('Started data inserting into DB')
        file_names = os.listdir(self.cfg['ftp']['localFolders']['tmp'])
        dps = []

        # Define artsig_inputs dictionary to collect inputs for the artificial features
        artsig_inputs = dict()

        for file_name in file_names:
            file_path = '%s/%s' % (self.cfg['ftp']['localFolders']['tmp'], file_name)
            self.logger.info('Getting data from %s' % file_path)

            # The signal is related to the entire Ticino canton (currently only RHW signal is handled)
            if 'VOPA' in file_name:
                dps = self.global_signals_handling(file_path, dps)

            # The signal is related to a specific location (belonging either to OASI or to ARPA domains)
            else:
                dps, artsig_inputs = self.location_signals_handling(file_path, file_name, dps, artsig_inputs)

        # todo for Dario: in artsig_inputs there should be everything you need to create the artificial features insert
        #  here the calculations

        # Send remaining points to InfluxDB
        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    def global_signals_handling(self, file_path, dps):
        f = open(file_path, 'rb')

        # cycling over the data
        for raw_row in f:
            # decode raw bytes
            row = raw_row.decode(constants.ENCODING)

            # Check the row considering the configured global signals
            for global_signal in self.cfg['globalSignals']:

                # Check if the row contains a RHW value
                if global_signal in row:
                    # discard final \r and \n characters
                    row = row[:-1]

                    # get data array
                    data = row.replace(' ', '').split('|')

                    # timestamp management
                    naive_time = datetime.strptime(data[1], '%Y%m%d000000')
                    local_dt = self.tz_local.localize(naive_time)
                    # add an hour to delete the DST influence (OASI timestamps are always in UTC+1 format)
                    utc_dt = local_dt.astimezone(pytz.utc) + local_dt.dst()

                    point = {
                        'time': int(utc_dt.timestamp()),
                        'measurement': self.cfg['influxDB']['measurementGlobal'],
                        'fields': dict(value=float(data[3])),
                        'tags': dict(signal=global_signal)
                    }
                    dps.append(copy.deepcopy(point))

        return dps

    def location_signals_handling(self, file_path, file_name, dps, artsig_inputs):
        # Open the file
        f = open(file_path, 'rb')

        # Read the first line
        raw = f.readline()

        # Check the datasets cases
        if 'arpa' in file_name:
            # ARPA case
            str_locations = raw.decode(constants.ENCODING)
            arpa_locations_keys = str_locations[:-1].split(';')[1:]
            measurement = self.cfg['influxDB']['measurementARPA']
        else:
            if 'meteosvizzera' in file_name:
                # MeteoSuisse case
                [key1, key2, _, _] = file_name.split('-')
                oasi_location_key = '%s-%s' % (key1, key2)
                measurement = self.cfg['influxDB']['measurementMeteoSuisse']
            else:
                # OASI case
                [oasi_location_key, _, _] = file_name.split('-')
                measurement = self.cfg['influxDB']['measurementOASI']

        # Signals
        raw = f.readline()
        str_signals = raw.decode(constants.ENCODING)
        signals = str_signals[:-1].split(';')[1:]

        # measure units, not used
        f.readline()

        # cycling over the data
        for raw_row in f:
            # decode raw bytes
            row = raw_row.decode(constants.ENCODING)
            row = row[:-1]

            # get raw date time
            dt = row.split(';')[0]

            # get data array
            data = row.split(';')[1:]

            # timestamp management
            naive_time = datetime.strptime(dt, self.cfg['local']['timeFormatMeasures'])
            local_dt = self.tz_local.localize(naive_time)
            # add an hour to delete the DST influence (OASI timestamps are always in UTC+1 format)
            utc_dt = local_dt.astimezone(pytz.utc) + local_dt.dst()

            # calculate the UTC timestamp
            utc_ts = int(utc_dt.timestamp())

            for i in range(0, len(data)):

                # Currently, the status are not taken into account
                if self.is_float(data[i]) is True and signals[i] != 'status':
                    if 'arpa' in file_name:
                        location_tag = constants.LOCATIONS[arpa_locations_keys[i]]
                    else:
                        location_tag = constants.LOCATIONS[oasi_location_key]

                    point = {
                        'time': utc_ts,
                        'measurement': measurement,
                        'fields': dict(value=float(data[i])),
                        'tags': dict(signal=signals[i], location=location_tag, case='MEASURED')
                    }
                    dps.append(copy.deepcopy(point))

                    # Check if this measure refers to a signal needed to calculate the artificial features
                    if signals[i] in self.cfg['artificialFeatures']['inputs']:
                        if (location_tag, signals[i]) not in artsig_inputs.keys():
                            artsig_inputs[(location_tag, signals[i])] = [point]
                        else:
                            artsig_inputs[(location_tag, signals[i])].append(point)

                    dps = self.point_handling(dps, point)

        return dps, artsig_inputs

    @staticmethod
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def save_measures_data(self, input_folder):
        """
        Save measures data (OASI and ARPA) in InfluxDB

        :param input_folder: input folder where the files are stored
        :type input_folder: string
        """
        self.logger.info('Started data inserting into DB')
        file_names = os.listdir(input_folder)
        dps = []

        tz_local = pytz.timezone(self.cfg['local']['timeZone'])

        for file_name in file_names:
            file_path = '%s/%s' % (input_folder, file_name)
            self.logger.info('Getting data from %s' % file_path)

            f = open(file_path, 'rb')

            # check ARPA datasets
            raw = f.readline()
            if 'arpa' in file_name:
                str_locations = raw.decode(constants.ENCODING)
                arpa_locations_keys = str_locations[:-1].split(';')[1:]
                measurement = self.cfg['influxDB']['measurementARPA']
            else:
                # get location code
                # MeteoSvizzera case
                if 'meteosvizzera' in file_name:
                    [key1, key2, date, stuff] = file_name.split('-')
                    oasi_location_key = '%s-%s' % (key1, key2)
                # OASI case
                else:
                    [oasi_location_key, date, stuff] = file_name.split('-')
                measurement = self.cfg['influxDB']['measurementOASI']

            # signals
            raw = f.readline()
            str_signals = raw.decode(constants.ENCODING)
            signals = str_signals[:-1].split(';')[1:]

            # measure units, not used
            f.readline()

            # cycling over the data
            for raw_row in f:
                # decode raw bytes
                row = raw_row.decode(constants.ENCODING)
                row = row[:-1]

                # get raw date time
                dt = row.split(';')[0]

                # get data array
                data = row.split(';')[1:]

                # timestamp management
                naive_time = datetime.strptime(dt, self.cfg['local']['timeFormatMeasures'])
                local_dt = tz_local.localize(naive_time)
                # add an hour to delete the DST influence (OASI timestamps are always in UTC+1 format)
                utc_dt = local_dt.astimezone(pytz.utc) + local_dt.dst()

                # calculate the UTC timestamp
                utc_ts = int(utc_dt.timestamp())

                for i in range(0, len(data)):

                    # Currently, the status are not taken into account
                    if self.is_float(data[i]) is True and signals[i] != 'status':
                        if 'arpa' in file_name:
                            location_tag = constants.LOCATIONS[arpa_locations_keys[i]]
                        else:
                            location_tag = constants.LOCATIONS[oasi_location_key]

                        point = {
                                    'time': utc_ts,
                                    'measurement': measurement,
                                    'fields': dict(value=float(data[i])),
                                    'tags': dict(signal=signals[i], location=location_tag, case='MEASURED')
                                }
                        dps = self.point_handling(dps, point)
            # close and delete the file
            f.close()
            os.unlink(file_path)

        # Send remaining points to InfluxDB
        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    def save_forecasts_data(self, input_folder):
        """
        Save measures data (OASI and ARPA) in InfluxDB

        :param input_folder: input folder where the files are stored
        :type input_folder: string
        """
        self.logger.info('Started data inserting into DB')
        file_names = os.listdir(input_folder)
        dps = []

        for file_name in file_names:
            file_path = '%s/%s' % (input_folder, file_name)
            self.logger.info('Getting data from %s' % file_path)
            (code, dt_code, ext) = file_path.split('.')

            f = open(file_path, 'rb')

            # VNXA51 case: stations forecasts
            if 'VNXA51' in code:
                dt_run = None
                signals = None

                for raw_row in f:
                    row = raw_row.decode(constants.ENCODING)
                    row = row[:-1]

                    # check if the signal dict is defined
                    if signals is not None:
                        data = row.split(';')
                        # set the running time
                        if dt_run is None:
                            dt_run = data[1]

                        for i in range(3, len(data)-1):
                            utc_time = datetime.strptime(dt_run, self.cfg['local']['timeFormatForecasts'])
                            utc_dt = pytz.utc.localize(utc_time)

                            # The Meteosuisse temperatures are in Kelvin degrees
                            if signals[i] in ['TD_2M', 'T_2M']:
                                val = float(data[i]) - 273.1
                            else:
                                val = float(data[i])

                            colon = data[2].find(':')

                            point = {
                                        'time': int(utc_dt.timestamp()),
                                        'measurement': self.cfg['influxDB']['measurementMeteoSuisse'],
                                        'fields': dict(value=val),
                                        'tags': dict(signal=signals[i], location=data[0], step='step%s' % data[2][colon-2:colon]) # hotfix DM 16.09.20
                                    }
                            dps = self.point_handling(dps, point)

                    # define signals
                    if row[0:3] == 'stn':
                        signals = row.split(';')
                        # skip the line related to unit measures
                        f.readline()

            # VOPA45 case: global forecast about meteorological situation in Ticino
            elif 'VOPA45' in code:
                for raw_row in f:
                    row = raw_row.decode(constants.ENCODING)
                    row = row[:-1]
                    if 'RHW' in row:
                        row = row.replace(' ', '')
                        (id, day, signal, value) = row.split('|')

                        utc_day = datetime.strptime(day, self.cfg['local']['timeFormatGlobal'])
                        utc_day = pytz.utc.localize(utc_day)

                        point = {
                                    'time': int(utc_day.timestamp()),
                                    'measurement': self.cfg['influxDB']['measurementGlobal'],
                                    'fields': dict(value=float(value)),
                                    'tags': dict(signal=signal)
                                }
                        dps = self.point_handling(dps, point)

            # close and delete the file
            f.close()
            os.unlink(file_path)

        # Send remaining points to InfluxDB
        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    def point_handling(self, dps, point):
        """
        Add a point and (eventually) store the entire dataset in InfluxDB

        :param point: point to add
        :type point: dict
        :param dps: input data points
        :type dps: list
        :return: output data points
        :rtype: list
        """
        dps.append(point)
        if len(dps) >= int(self.cfg['influxDB']['maxLinesPerInsert']):
            self.logger.info('Sent %i points to InfluxDB server' % len(dps))
            self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])
            dps = []
            time.sleep(0.1)
        return dps

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

            # Update the training datasets
            self.update_training_datasets(new_value=daily_max, location=series['tags']['location'])

        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    def update_training_datasets(self, new_value, location):
        """
        Update the output training dataset
        :param new_value: new output value
        :type new_value: float
        :param location: location related to the new value
        :type location: string
        """
        # define the files paths
        inputs_tmp_file = '%s/tmp/%s_%s.csv' % (self.cfg['local']['trainingDatasets'], location, self.forecast_type)
        inputs_file = '%s/inputs/%s_%s.csv' % (self.cfg['local']['trainingDatasets'], location, self.forecast_type)
        outputs_file = '%s/outputs/%s_%s.csv' % (self.cfg['local']['trainingDatasets'], location, self.forecast_type)

        # check if the temporary inputs file exists
        yesterday = datetime.strftime(datetime.now() - timedelta(days=1), '%Y-%m-%d')
        if os.path.isfile(inputs_tmp_file):

            # get the last data appended to the inputs dataset
            data_tail = tailhead.tail(open(inputs_tmp_file, 'rb'), 1)
            last_data_inputs_tmp = data_tail[0].decode('utf-8')

            # get the last data appended to the input dataset
            data_tail = tailhead.tail(open(inputs_file, 'rb'), 1)
            last_data_inputs = data_tail[0].decode('utf-8')

            # get the last data appended to the output dataset
            data_tail = tailhead.tail(open(outputs_file, 'rb'), 1)
            last_data_outputs = data_tail[0].decode('utf-8')

            # check if the last inputs string in the temporary file is related to yesterday and
            # no data about yesterday were already saved in the datasets
            if yesterday in last_data_inputs_tmp and yesterday not in last_data_inputs and \
               yesterday not in last_data_outputs:

                # append inputs
                with open(inputs_file, 'a') as fw:
                    fw.write('%s\n' % last_data_inputs_tmp)
                fw.close()

                # append outputs
                with open(outputs_file, 'a') as fw:
                    fw.write('%s,%.3f\n' % (yesterday, new_value))
                fw.close()

            # delete the temporary file
            os.unlink(inputs_tmp_file)
        else:
            self.logger.warning('No inputs available for station %s, case %s, day %s' % (location, self.forecast_type,
                                                                                         yesterday))

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

    @staticmethod
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def get_previous_day(self):
        if self.cfg['forecastPeriod']['case'] == 'current':
            dt = datetime.now(pytz.utc) - timedelta(1)
        else:
            dt = datetime.strptime(self.cfg['dayToForecast'], '%Y-%m-%d')
            dt = pytz.utc.localize(dt)
            dt = dt - timedelta(1)
        return dt

    def calc_kpis(self):
        # get the date to analyze
        dt = self.get_previous_day()
        dt = dt.replace(hour=0, minute=0, second=0)

        # cycle over the starting dates
        for start_date in self.cfg['kpis']['startingDates']:
            dt_start_date = datetime.strptime(start_date, '%Y-%m-%d')
            dt_start_date = pytz.utc.localize(dt_start_date)

            end_date = dt.strftime('%Y-%m-%d')
            # end_date = '2019-06-15'

            # get measurement data
            query = 'SELECT mean(value) FROM %s WHERE signal=\'YO3\' AND time>=\'%sT00:00:00Z\' AND ' \
                    'time<=\'%sT23:59:59Z\' GROUP BY time(1d), location, signal' % (self.cfg['influxDB']['measurementOASI'],
                                                                                    start_date,
                                                                                    end_date)

            self.logger.info('Performing query: %s' % query)
            res = self.influxdb_client.query(query, epoch='s')

            dfs_meas = dict()
            for series in res.raw['series']:
                ts = []
                vals = []
                for i in range(0, len(series['values'])):
                    if series['values'][i][1] is not None:
                        ts.append(series['values'][i][0])
                        vals.append(series['values'][i][1])
                df = pd.DataFrame.from_dict({'ts': ts, series['tags']['location']: vals})
                df = df.set_index('ts')
                dfs_meas[series['tags']['location']] = df

            # get forecasts data
            query = 'SELECT mean(forecast), mean(forecastRF) FROM %s WHERE time>=\'%sT00:00:00Z\' AND time<=\'%sT23:59:59Z\' ' \
                    'GROUP BY time(1d), location, case, predictor' % (self.cfg['influxDB']['measurementForecasts'],
                                                                      start_date, end_date)

            self.logger.info('Performing query: %s' % query)
            res = self.influxdb_client.query(query, epoch='s')

            dfs_forecasts_ens = dict()
            dfs_forecasts_rf = dict()
            for series in res.raw['series']:
                ts = []
                vals_ens = []
                vals_rf = []
                for i in range(0, len(series['values'])):
                    if series['values'][i][1] is not None:
                        ts.append(series['values'][i][0])
                        vals_ens.append(series['values'][i][1])
                        vals_rf.append(series['values'][i][2])

                id = '%s__%s__%s' % (series['tags']['location'], series['tags']['case'], series['tags']['predictor'])

                df_ens = pd.DataFrame.from_dict({'ts': ts, id: vals_ens})
                df_ens = df_ens.set_index('ts')
                dfs_forecasts_ens[id] = df_ens

                df_rf = pd.DataFrame.from_dict({'ts': ts, id: vals_rf})
                df_rf = df_rf.set_index('ts')
                dfs_forecasts_rf[id] = df_rf

            # calculate the kpis
            dps = []
            for id in dfs_forecasts_ens.keys():
                (location, case, predictor) = id.split('__')

                # merge the dataframes
                mdf_ens = pd.concat([dfs_forecasts_ens[id], dfs_meas[location]], axis=1, join='inner')
                mdf_rf = pd.concat([dfs_forecasts_rf[id], dfs_meas[location]], axis=1, join='inner')

                # additional simple checking on the datasets
                if len(mdf_ens[location]) == len(mdf_ens[id]) and len(mdf_rf[location]) == len(mdf_rf[id]):
                    rmse_ens = np.sqrt(metrics.mean_squared_error(mdf_ens[location], mdf_ens[id]))
                    mae_ens = metrics.mean_absolute_error(mdf_ens[location], mdf_ens[id])

                    point = {
                                'time': int(dt_start_date.timestamp()),
                                'measurement': self.cfg['influxDB']['measurementForecastsKPIs'],
                                'fields': dict(rmse=float(rmse_ens), mae=float(mae_ens)),
                                'tags': dict(location=location, case=case, predictor=predictor,
                                             start_date='sd_%s' % start_date, type='ENS')
                            }
                    dps = self.point_handling(dps, point)

                    rmse_rf = np.sqrt(metrics.mean_squared_error(mdf_rf[location], mdf_rf[id]))
                    mae_rf = metrics.mean_absolute_error(mdf_rf[location], mdf_rf[id])

                    point = {
                                'time': int(dt_start_date.timestamp()),
                                'measurement': self.cfg['influxDB']['measurementForecastsKPIs'],
                                'fields': dict(rmse=float(rmse_rf), mae=float(mae_rf)),
                                'tags': dict(location=location, case=case, predictor=predictor,
                                             start_date='sd_%s' % start_date, type='RF')
                            }
                    dps = self.point_handling(dps, point)

                    # calculate the errors
                    mdf_ens['abs_err'] = abs(mdf_ens[id] - mdf_ens[location])
                    mdf_ens['err'] = mdf_ens[id] - mdf_ens[location]

                    mdf_rf['abs_err'] = abs(mdf_rf[id] - mdf_rf[location])
                    mdf_rf['err'] = mdf_rf[id] - mdf_rf[location]

                    # insert the errors data

                    # ensemble case
                    for index, row in mdf_ens.iterrows():
                        point = {
                                    'time': int(index),
                                    'measurement': self.cfg['influxDB']['measurementForecastsKPIs'],
                                    'fields': dict(err=float(row['err']), abs_err=float(row['abs_err'])),
                                    'tags': dict(location=location, case=case, predictor=predictor, type='ENS')}
                        dps = self.point_handling(dps, point)

                    # random forest
                    for index, row in mdf_rf.iterrows():
                        point = {
                                    'time': int(index),
                                    'measurement': self.cfg['influxDB']['measurementForecastsKPIs'],
                                    'fields': dict(err=float(row['err']), abs_err=float(row['abs_err'])),
                                    'tags': dict(location=location, case=case, predictor=predictor, type='RF')}
                        dps = self.point_handling(dps, point)
                else:
                    self.logger.info('Unable to calculate the kpis')

        # Send data points to InfluxDB
        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

