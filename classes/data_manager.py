# import section
import os
import datetime
import zipfile

import tailhead
import copy
import pytz
import time
import ftplib
import numpy as np
import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn.metrics as metrics
from zipfile import ZipFile

from datetime import date, datetime, timedelta
from influxdb import InfluxDBClient

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
        self.files_correctly_downloaded = []
        self.files_not_correctly_downloaded = dict()
        self.files_correctly_handled = []
        self.files_not_correctly_handled = dict()
        self.ftp = None

        # Define the time zone
        self.tz_local = pytz.timezone(self.cfg['local']['timeZone'])

    def open_ftp_connection(self):
        # perform FTP connection and login
        self.ftp = ftplib.FTP(self.cfg['ftp']['host'])
        self.ftp.login(self.cfg['ftp']['user'], self.cfg['ftp']['password'])

    def close_ftp_connection(self):
        # close the FTP connection
        self.ftp.close()

    def upload_file(self, file_to_send):
        curr_wd = os.getcwd()
        os.chdir('%s%s%s' % (os.getcwd(), os.sep, self.cfg['ftp']['localFolders']['tmp']))
        try:
            self.ftp.cwd(self.cfg['ftp']['remoteFolders']['results'])
            with open(file_to_send, 'rb') as f:
                self.ftp.storbinary('STOR %s' % file_to_send, f)
            os.chdir(curr_wd)
            return True

        except Exception as e:
            self.logger.error('Exception: %s' % str(e))
            os.chdir(curr_wd)
            return False

    def download_remote_files(self):

        if os.path.isdir(self.cfg['ftp']['localFolders']['tmp']) is False:
            os.mkdir(self.cfg['ftp']['localFolders']['tmp'])

        self.files_correctly_downloaded = []
        self.files_not_correctly_downloaded = dict()

        try:
            # todo Check if also the forecast folder is effectively used
            for ftp_dir in [self.cfg['ftp']['remoteFolders']['measures'],
                            self.cfg['ftp']['remoteFolders']['forecasts']]:
                self.logger.info('Getting files via FTP from %s/%s' % (self.cfg['ftp']['host'], ftp_dir))
                self.ftp.cwd('/%s' % ftp_dir)

                # cycle over the remote files
                for file_name in self.ftp.nlst('*'):
                    tmp_local_file = os.path.join(self.cfg['ftp']['localFolders']['tmp'] + "/", file_name)
                    try:
                        self.logger.info('%s/%s -> %s/%s/%s' % (ftp_dir, file_name, os.getcwd(),
                                                                self.cfg['ftp']['localFolders']['tmp'], file_name))

                        # get the file from the server
                        with open(tmp_local_file, 'wb') as f:
                            def callback(data):
                                f.write(data)

                            self.ftp.retrbinary("RETR " + file_name, callback)

                        self.files_correctly_downloaded.append(file_name)
                    except Exception as e:
                        self.logger.error('Downloading exception: %s' % str(e))
                        self.files_not_correctly_downloaded[file_name] = str(e)
        except Exception as e:
            self.logger.error('Connection exception: %s' % str(e))


    def delete_remote_files(self):

        if os.path.isdir(self.cfg['ftp']['localFolders']['tmp']) is False:
            os.mkdir(self.cfg['ftp']['localFolders']['tmp'])

        try:
            # cycle over the remote files
            for file_to_delete in self.files_correctly_handled:
                # Set the remote folder
                if 'VOPA' in file_to_delete or 'VNXA51' in file_to_delete:
                    self.ftp.cwd('/%s' % self.cfg['ftp']['remoteFolders']['forecasts'])
                else:
                    self.ftp.cwd('/%s' % self.cfg['ftp']['remoteFolders']['measures'])

                try:
                    # delete the remote file
                    self.logger.info('Delete remote file %s' % file_to_delete)
                    self.ftp.delete(file_to_delete)
                except Exception as e:
                    self.logger.error('Unable to delete remote file %s' % file_to_delete)
                    self.logger.error('Exception: %s' % str(e))
        except Exception as e:
            self.logger.error('Connection exception: %s' % str(e))

    def insert_data(self):
        self.logger.info('Started data inserting into DB')
        file_names = os.listdir(self.cfg['ftp']['localFolders']['tmp'])
        dps = []

        self.files_correctly_handled = []
        self.files_not_correctly_handled = dict()
        for file_name in file_names:
            file_path = '%s/%s' % (self.cfg['ftp']['localFolders']['tmp'], file_name)
            self.logger.info('Getting data from %s' % file_path)

            try:
                # Meteosuisse forecasts
                if 'VOPA' in file_name or 'VNXA51' in file_name:
                    dps = self.handle_meteo_forecasts(file_path, dps)

                # OASI/ARPA/Meteosuisse measurements
                else:
                    dps = self.location_signals_handling(file_path, file_name, dps)

                # Archive file
                self.archive_file(file_name)
                self.files_correctly_handled.append(file_name)
            except Exception as e:
                self.logger.error('EXCEPTION: %s' % str(e))
                self.files_not_correctly_handled[file_name] = str(e)
                # Delete the raw file
                os.unlink('%s%s%s' % (self.cfg['ftp']['localFolders']['tmp'], os.sep, file_name))

        # Send remaining points to InfluxDB
        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    def archive_file(self, file_name):
        if 'VOPA' in file_name or 'VNXA51' in file_name:
            archive_folder = '%s%s%s' % (self.cfg['ftp']['localFolders']['archive'], os.sep, file_name.split('.')[1][0:8])
            self.check_folder(archive_folder)
        else:
            if 'meteosvizzera' in file_name or 'arpal' in file_name or 'nabel' in file_name:
                archive_folder = '%s%s%s' % (self.cfg['ftp']['localFolders']['archive'], os.sep, file_name.split('-')[2])
                self.check_folder(archive_folder)
            else:
                archive_folder = '%s%s%s' % (self.cfg['ftp']['localFolders']['archive'], os.sep, file_name.split('-')[1])
                self.check_folder(archive_folder)

        # Zip the raw file
        zip_obj = ZipFile('%s%s%s.zip' % (archive_folder, os.sep, file_name), 'w')
        zip_obj.write(('%s%s%s' % (self.cfg['ftp']['localFolders']['tmp'], os.sep, file_name)),
                      compress_type=zipfile.ZIP_DEFLATED, arcname=file_name)
        zip_obj.close()

        # Delete the raw file
        os.unlink('%s%s%s' % (self.cfg['ftp']['localFolders']['tmp'], os.sep, file_name))

    @staticmethod
    def check_folder(folder):
        if os.path.exists(folder) is False:
            os.mkdir(folder)

    def location_signals_handling(self, file_path, file_name, dps):
        # Open the file
        f = open(file_path, 'rb')

        # Read the first line
        raw = f.readline()

        # Check the datasets cases, I apologize for the hard-coding
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
                if 'nabel' in file_name:
                    # Nabel OASI case
                    [key1, key2, _, _] = file_name.split('-')
                    oasi_location_key = '%s-%s' % (key1, key2)
                else:
                    # Simple OASI case
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
                    dps = self.point_handling(dps, point)

        return dps

    @staticmethod
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def handle_meteo_forecasts(self, input_file, dps):
        """
        Save measures data (OASI and ARPA) in InfluxDB

        :param input_file: input file
        :type input_file: string
        """

        self.logger.info('Getting data from %s' % input_file)
        (code, dt_code, ext) = input_file.split('.')

        f = open(input_file, 'rb')

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

                        # Find the : position
                        colon = data[2].find(':')

                        point = {
                                    'time': int(utc_dt.timestamp()),
                                    'measurement': self.cfg['influxDB']['measurementMeteoSuisse'],
                                    'fields': dict(value=val),
                                    'tags': dict(signal=signals[i], location=data[0], step='step%s' % data[2][colon-2:colon])
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
                    dps.append(copy.deepcopy(point))

        # close the file
        f.close()
        # os.unlink(file_path)

        return dps

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
        dps.append(copy.deepcopy(point))
        if len(dps) >= int(self.cfg['influxDB']['maxLinesPerInsert']):
            self.logger.info('Sent %i points to InfluxDB server' % len(dps))
            self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])
            dps = []
            time.sleep(0.1)
        return dps

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
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

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

