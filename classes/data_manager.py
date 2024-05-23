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

    def __init__(self, influxdb_client, cfg, logger, influx_df_client=None):
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
        self.influx_df_client = influx_df_client
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

    @staticmethod
    def is_meteosuisse_forecast_file(file_name):
        if ('VNXA51' in file_name or 'VQCA19' in file_name or 'VNYA34' in file_name or 'VNYA32' in file_name or
                'VNXA54' in file_name or 'icon' in file_name):
            return True
        else:
            return False

    def delete_remote_files(self):

        if os.path.isdir(self.cfg['ftp']['localFolders']['tmp']) is False:
            os.mkdir(self.cfg['ftp']['localFolders']['tmp'])

        try:
            # cycle over the remote files
            for file_to_delete in self.files_correctly_handled:
                # Set the remote folder
                if self.is_meteosuisse_forecast_file(file_to_delete):
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
                # Meteo forecasts provided by Meteosuisse
                if self.is_meteosuisse_forecast_file(file_name):
                    dps = self.handle_meteo_forecasts_file(file_path, dps)

                # OASI/ARPA/Meteosuisse measurements
                else:
                    dps = self.handle_measures_file(file_path, file_name, dps)

                # Archive file
                self.archive_file(file_name)
                self.files_correctly_handled.append(file_name)
            except Exception as e:
                self.logger.error('EXCEPTION: %s, file %s' % (str(e), file_name))
                self.files_not_correctly_handled[file_name] = str(e)
                # Delete the raw file
                os.unlink('%s%s%s' % (self.cfg['ftp']['localFolders']['tmp'], os.sep, file_name))

        # Send remaining points to InfluxDB
        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

    def archive_file(self, file_name):
        if self.is_meteosuisse_forecast_file(file_name):
            archive_folder = '%s%s%s' % (
            self.cfg['ftp']['localFolders']['archive'], os.sep, file_name.split('.')[1][0:8])
            self.check_folder(archive_folder)
        else:
            if 'meteosvizzera' in file_name or 'arpal' in file_name or 'nabel' in file_name:
                archive_folder = '%s%s%s' % (
                self.cfg['ftp']['localFolders']['archive'], os.sep, file_name.split('-')[2])
                self.check_folder(archive_folder)
            else:
                archive_folder = '%s%s%s' % (
                self.cfg['ftp']['localFolders']['archive'], os.sep, file_name.split('-')[1])
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

    def handle_measures_file(self, file_path, file_name, dps):
        # Open the file
        f = open(file_path, 'rb')

        # Read the first line
        raw = f.readline()

        # Check the datasets cases, I apologize for the hard-coding
        if 'arpa' in file_name:
            # ARPA case
            str_locations = raw.decode(constants.ENCODING)
            arpa_locations_keys = str_locations[:-1].split(';')[1:]
            measurement = self.cfg['influxDB']['measurementInputsMeasurements']
        else:
            if 'meteosvizzera' in file_name:
                # MeteoSuisse case
                [key1, key2, _, _] = file_name.split('-')
                oasi_location_key = '%s-%s' % (key1, key2)
                measurement = self.cfg['influxDB']['measurementInputsMeasurements']
            else:
                if 'nabel' in file_name:
                    # Nabel OASI case
                    [key1, key2, _, _] = file_name.split('-')
                    oasi_location_key = '%s-%s' % (key1, key2)
                else:
                    # Simple OASI case
                    [oasi_location_key, _, _] = file_name.split('-')
                measurement = self.cfg['influxDB']['measurementInputsMeasurements']

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

    def handle_meteo_forecasts_file(self, input_file, dps):
        """
        Save measures data (OASI and ARPA) in InfluxDB

        :param input_file: input file
        :type input_file: string
        """
        if 'icon' in input_file:
            self.logger.info('Getting ICON data from file %s' % input_file)
            (prefix, code, suffix) = input_file.split(os.sep)[-1].split('-')
        else:
            self.logger.info('Getting COSMO data from file %s' % input_file)
            prefix = 'cosmo'
            (code, dt_code, ext) = input_file.split('.')

        f = open(input_file, 'rb')

        # Single predition (COSMO codes: VNXA51 -> COSMO1, VNYA34 -> COSMO2)
        if ('VNYA34' in code or 'VNXA51' in code or             # COSMO case
           (prefix == 'icon' and 'profile' not in suffix)):     # ICON case

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

                    for i in range(3, len(data) - 1):
                        utc_time = datetime.strptime(dt_run, self.cfg['local']['timeFormatForecasts'])
                        utc_dt = pytz.utc.localize(utc_time)

                        # The Meteosuisse temperatures are in Kelvin degrees
                        if signals[i] in ['TD_2M', 'T_2M']:
                            val = float(data[i]) - 273.1
                        else:
                            val = float(data[i])

                        # Check if the data is available
                        if val != -99999:
                            # Find the : position
                            colon = data[2].find(':')

                            # Step and COSMO2 suffix handling
                            step = int(data[2].split(':')[0])
                            #   COSMOS case         ICON case
                            if 'VNYA34' in code or code == 'ch2':
                                str_step = 'step%03d' % step
                                signal_suffix = '_c2'
                            else:
                                str_step = 'step%02d' % step
                                signal_suffix = ''

                            point = {
                                'time': int(utc_dt.timestamp()),
                                'measurement': self.cfg['influxDB']['measurementInputsForecasts'],
                                'fields': dict(value=val),
                                'tags': dict(signal='%s%s' % (signals[i], signal_suffix), location=data[0],
                                             step=str_step)
                            }
                            dps = self.point_handling(dps, point)

                # define signals
                if row[0:3] == 'stn':
                    signals = row.split(';')
                    # skip the line related to unit measures
                    f.readline()

        # Vertical gradients (COSMO codes: VNXA54 -> COSMO1, VNYA32 -> COSMO2)
        elif (('VNYA32' in code or 'VNXA54' in code) or         # COSMO
              (prefix == 'icon' and 'profile' in suffix)):      # ICON
            dt_run = None

            single_sigs = self.get_single_signals(f)

            for raw_row in f:
                row = raw_row.decode(constants.ENCODING)
                row = row.replace('\n', '')
                row = row[:-1]
                data = row.split(';')

                # set the running time
                if dt_run is None:
                    dt_run = data[1]

                for i in range(3, len(data)):
                    utc_time = datetime.strptime(dt_run, self.cfg['local']['timeFormatForecasts'])
                    utc_dt = pytz.utc.localize(utc_time)

                    # Step and COSMO2 suffix handling
                    step = int(data[2].split(':')[0])
                    #   COSMOS case         ICON case
                    if 'VNYA32' in code or code == 'ch2':
                        str_step = 'step%03d' % step
                        signal_suffix = '_c2'
                    else:
                        str_step = 'step%02d' % step
                        signal_suffix = ''

                    point = {
                        'time': int(utc_dt.timestamp()),
                        'measurement': self.cfg['influxDB']['measurementInputsForecasts'],
                        'fields': dict(value=float(data[i])),
                        'tags': dict(signal='%s%s' % (single_sigs[i-3], signal_suffix), location=data[0],
                                     step=str_step)
                    }
                    dps = self.point_handling(dps, point)

        # VQCA19 case: global forecast about meteorological situation in Ticino (available only for COSMO)
        elif 'VQCA19' in code:
            for raw_row in f:
                row = raw_row.decode(constants.ENCODING)
                row = row[:-1]
                if 'RHW' in row:
                    # row = row.replace(' ', '')
                    (signal, day, value) = row.replace('  ', ',').replace(' ', '').replace(',,', ',')[0:-2].split(',')

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

    def get_single_signals(self, fr):
        str_sigs = False
        str_quotes = False
        while str_sigs is False or str_quotes is False:
            raw_row = fr.readline()
            row = raw_row.decode(constants.ENCODING)
            row = row.replace('\n', '')
            row = row[:-1]

            if 'stn;' in row:
                str_sigs = row
            if ';level;' in row:
                str_quotes = row

        sigs = str_sigs.split(';')
        quotes = str_quotes.split(';')

        if len(sigs) == len(quotes):
            new_sigs = []
            for i in range(3, len(sigs)):
                new_sigs.append('%s%s' % (sigs[i], quotes[i]))
        return new_sigs

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
                    'time<=\'%sT23:59:59Z\' GROUP BY time(1d), location, signal' % (
                    self.cfg['influxDB']['measurementOASI'],
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

    def get_start_end_dates(self):
        if self.cfg['calculatedInputsSection']['period'] == 'custom':
            start_date = self.cfg['calculatedInputsSection']['from']
            end_date = self.cfg['calculatedInputsSection']['to']
        else:
            last_hours = int(self.cfg['calculatedInputsSection']['period'].replace('last', '').replace('h', ''))
            start_date = (datetime.now() - timedelta(hours=last_hours)).strftime('%Y-%m-%dT%H:00:00Z')
            end_date = datetime.now().strftime('%Y-%m-%dT%H:59:59Z')
        return start_date, end_date

    def calculate_artificial_data(self):
        if self.cfg['intervalSettings']['period'] == 'custom':
            start_day = datetime.strptime(self.cfg['intervalSettings']['from'], '%Y-%m-%d')
            end_day = datetime.strptime(self.cfg['intervalSettings']['to'], '%Y-%m-%d')

            # Cycle over the days
            curr_day = start_day
            while curr_day <= end_day:
                curr_str = curr_day.strftime('%Y-%m-%d')
                self.calc_artificial_data_for_given_period(curr_str, curr_str)
                curr_day += timedelta(days=1)
        else:
            last_hours = int(self.cfg['intervalSettings']['period'].replace('last', '').replace('h', ''))
            start_str = (datetime.now() - timedelta(hours=last_hours)).strftime('%Y-%m-%d')
            end_str = datetime.now().strftime('%Y-%m-%d')
            self.calc_artificial_data_for_given_period(start_str, end_str)

    def create_aggregated_data(self):
        if self.cfg['intervalSettings']['period'] == 'custom':
            start_day = datetime.strptime(self.cfg['intervalSettings']['from'], '%Y-%m-%d')
            end_day = datetime.strptime(self.cfg['intervalSettings']['to'], '%Y-%m-%d')

            # Cycle over the days
            curr_day = start_day
            while curr_day <= end_day:
                curr_str = curr_day.strftime('%Y-%m-%d')
                self.calc_aggregated_data_for_given_period(curr_str, curr_str)
                curr_day += timedelta(days=1)
        else:
            last_hours = int(self.cfg['intervalSettings']['period'].replace('last', '').replace('h', ''))
            start_str = (datetime.now() - timedelta(hours=last_hours)).strftime('%Y-%m-%d')
            end_str = datetime.now().strftime('%Y-%m-%d')
            self.calc_aggregated_data_for_given_period(start_str, end_str)

    def calc_artificial_data_for_given_period(self, from_str, to_str):
        self.logger.info('Calculate artificial data for period [%s:%s]' % (from_str, to_str))

        # To be safe go from the beginning of the "from" date to the end of "to" date
        from_str = '%sT00:00:00Z' % from_str
        to_str = '%sT23:59:59Z' % to_str

        for asig_data in self.cfg['calculatedInputsSection']['inputs']:
            try:
                if asig_data['forecast'] is False:
                    input1 = self.get_measure_data(asig_data['locations'][0], asig_data['signals'][0], from_str, to_str)
                    input2 = self.get_measure_data(asig_data['locations'][1], asig_data['signals'][1], from_str, to_str)
                    tag_loc, tag_sig = self.get_loc_sig(asig_data['locations'], asig_data['signals'], asig_data['function'])

                    # Check if input data are meaningful
                    if input1 is not None and input2 is not None and input1.index.equals(input2.index) is True:
                        output = self.apply_function_to_measures(asig_data['function'], input1, input2)
                        # Apply gain and offset
                        output['value'] = output['value'] * asig_data['gain'] + asig_data['offset']

                        if asig_data['booleanOutputThreshold'] is not False:
                            output = self.digitalize_output(output, asig_data['booleanOutputThreshold'])
                            tag_sig = 'B%s' % tag_sig

                        res = self.influx_df_client.write_points(output,
                                                                 self.cfg['influxDB']['measurementInputsMeasurements'],
                                                                 tags={'location': tag_loc, 'signal': tag_sig},
                                                                 protocol='line')
                        if res is not True:
                            self.logger.warning('Failed measure inserting for couple [%s:%s]' % (tag_loc, tag_sig))
                        else:
                            self.logger.info('Inserted measure data for couple [%s:%s]' % (tag_loc, tag_sig))
                    else:
                        self.logger.warning('Data missing or index mismatch: measure not inserted for couple [%s:%s]' % (tag_loc, tag_sig))
                else:
                    input1 = self.get_forecast_data(asig_data['locations'][0], asig_data['signals'][0], from_str, to_str)
                    input2 = self.get_forecast_data(asig_data['locations'][1], asig_data['signals'][1], from_str, to_str)
                    tag_loc, tag_sig = self.get_loc_sig(asig_data['locations'], asig_data['signals'], asig_data['function'])

                    # Check if input data are meaningful
                    if input1 is not None and input2 is not None and input1.index.equals(input2.index) is True:
                        output = self.apply_function_to_forecast(asig_data['function'], input1, input2)
                        # Apply gain and offset
                        output['value'] = output['value'] * asig_data['gain'] + asig_data['offset']

                        res = self.influx_df_client.write_points(output, self.cfg['influxDB']['measurementInputsForecasts'],
                                                                 tags={'location': tag_loc, 'signal': tag_sig},
                                                                 tag_columns=['step'], protocol='line')
                        if res is not True:
                            self.logger.warning('Failed forecast inserting for couple [%s:%s]' % (tag_loc, tag_sig))
                        else:
                            self.logger.info('Inserted forecast data for couple [%s:%s]' % (tag_loc, tag_sig))
                    else:
                        self.logger.warning('Data missing or index mismatch: forecast not inserted for couple [%s:%s]' % (tag_loc, tag_sig))
            except Exception as e:
                self.logger.error('EXCEPTION: %s' % str(e))
                self.logger.info('Failed data calculation for couple [%s:%s]' % (tag_loc, tag_sig))

    def calc_aggregated_data_for_given_period(self, from_str, to_str):
        self.logger.info('Created aggregated data for period [%s:%s]' % (from_str, to_str))

        # To be safe go from the beginning of the "from" date to the end of "to" date
        from_str = '%sT00:00:00Z' % from_str
        to_str = '%sT23:59:59Z' % to_str

        for asig_data in self.cfg['aggregatedInputsSection']['inputs']:
            try:
                if asig_data['forecast'] is False:
                    inputs = []
                    for i in range(0, len(asig_data['locations'])):
                        tmp_df = self.get_measure_data(asig_data['locations'][i], asig_data['signals'][i], from_str, to_str)
                        tmp_df = tmp_df.rename(columns={'value': 'value_%i' % i})
                        inputs.append(tmp_df)
                    input_df = pd.concat(inputs, axis=1, join="inner")

                    agg_df = self.aggregate_dataframes(input_df, asig_data['function'])

                    output = pd.concat([agg_df], axis=1)
                    output = output.rename(columns={0: 'value'})

                    res = self.influx_df_client.write_points(output,
                                                             self.cfg['influxDB']['measurementInputsMeasurements'],
                                                             tags={'location': asig_data['location'],
                                                                   'signal': asig_data['signal']},
                                                             protocol='line')
                    if res is not True:
                        self.logger.warning('Failed measure inserting for couple [%s:%s]' % (asig_data['location'],
                                                                                             asig_data['signal']))
                    else:
                        self.logger.info('Inserted measure data for couple [%s:%s]' % (asig_data['location'],
                                                                                       asig_data['signal']))
                else:
                    inputs = []
                    for i in range(0, len(asig_data['locations'])):
                        tmp_df = self.get_forecast_data(asig_data['locations'][i], asig_data['signals'][i], from_str, to_str)
                        if i == 0:
                            idx_step_df = tmp_df
                            idx_step_df = idx_step_df.drop(['value'], axis=1)

                        tmp_df = tmp_df.drop(['index', 'step'], axis=1)
                        tmp_df = tmp_df.rename(columns={'value': 'value_%i' % i})
                        inputs.append(tmp_df)

                    input_df = pd.concat(inputs, axis=1, join='inner')

                    agg_df = self.aggregate_dataframes(input_df, asig_data['function'])

                    output = pd.concat([agg_df, idx_step_df], axis=1)
                    output = output.rename(columns={0: 'value'})
                    output = output.set_index('index')
                    output = output.reindex(columns=['value', 'step'])

                    res = self.influx_df_client.write_points(output, self.cfg['influxDB']['measurementInputsForecasts'],
                                                             tags={'location': asig_data['location'],
                                                                   'signal': asig_data['signal']},
                                                             tag_columns=['step'], protocol='line')
                    if res is not True:
                        self.logger.warning('Failed forecast inserting for couple [%s:%s]' % (asig_data['location'],
                                                                                              asig_data['signal']))
                    else:
                        self.logger.info('Inserted forecast data for couple [%s:%s]' % (asig_data['location'],
                                                                                        asig_data['signal']))
            except Exception as e:
                self.logger.error('EXCEPTION: %s' % str(e))
                self.logger.info('Failed data calculation for couple [%s:%s]' % (asig_data['location'],
                                                                                 asig_data['signal']))

    @staticmethod
    def aggregate_dataframes(inputs, func):
        if func == 'max':
            return inputs.max(axis=1)
        elif func == 'mean':
            return inputs.mean(axis=1)
        elif func == 'min':
            return inputs.min(axis=1)

    @staticmethod
    def digitalize_output(output, pars):
        # Currently it works only with > and < operators
        op = pars[0]
        th = float(pars[1:])
        if op == '>':
            output['value'] = [1.0 if x > th else 0.0 for x in output['value']]
        elif op == '<':
            output['value'] = [1.0 if x < th else 0.0 for x in output['value']]
        return output

    def get_measure_data(self, loc, sig, st_date, end_date):
        query = 'SELECT mean(value) AS value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time>=\'%s\' AND ' \
                'time<=\'%s\' GROUP BY time(30m)' % (self.cfg['influxDB']['measurementInputsMeasurements'],
                                                     loc, sig, st_date, end_date)
        res = self.influx_df_client.query(query=query)
        if self.cfg['influxDB']['measurementInputsMeasurements'] in res.keys():
            return res[self.cfg['influxDB']['measurementInputsMeasurements']]
        else:
            return None

    def get_forecast_data(self, loc, sig, st_date, end_date):
        query = 'SELECT value, step FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time>=\'%s\' AND ' \
                'time<=\'%s\'' % (self.cfg['influxDB']['measurementInputsForecasts'], loc, sig, st_date, end_date)
        res = self.influx_df_client.query(query=query)
        df = res[self.cfg['influxDB']['measurementInputsForecasts']]
        df.reset_index(inplace=True)
        df['idx'] = df['index'].astype(str) + "-" + df['step'].astype(str)
        df = df.set_index('idx')
        return df

    @staticmethod
    def apply_function_to_measures(func, i1, i2):
        if func == 'sum':
            return i1+i2
        elif func == 'diff':
            return i1-i2
        elif func == 'mul':
            return i1*i2
        elif func == 'ratio':
            return i1/i2

    @staticmethod
    def apply_function_to_forecast(func, i1, i2):
        if func == 'sum':
            output_series = i1['value']+i2['value']
        elif func == 'diff':
            output_series = i1['value']-i2['value']
        elif func == 'mul':
            output_series = i1['value']*i2['value']
        elif func == 'ratio':
            output_series = i1['value']/i2['value']
        output = copy.deepcopy(i1)
        output['value'] = output_series
        return output.set_index('index')

    @staticmethod
    def get_loc_sig(locations, signals, func):
        return '%s_%s' % (locations[0], locations[1]), '%s_%s_%s' % (func[0].upper(), signals[0], signals[1])

