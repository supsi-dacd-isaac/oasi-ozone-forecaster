# import section
import os
import datetime
import glob
import ftplib
import pytz
import scipy.io as sio
import numpy as np

from datetime import datetime, timedelta
from glob import glob
from influxdb import InfluxDBClient

import constants

class ResultsManager:
    """
    Class to properly handle the forecasters results
    """

    def __init__(self, influxdb_client, cfg, logger, forecast_type, predicted_day):
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
        :param predicted_day: predicted day
        :type predicted_day: string
        """
        # set the variables
        self.influxdb_client = influxdb_client
        self.cfg = cfg
        self.logger = logger
        self.forecast_type = forecast_type
        self.predicted_day = predicted_day

    def handle(self):
        """
        Handle the results, i.e. save them into InfluxDB and send CSV resume on a remote FTP server
        """
        dt = datetime.now(pytz.utc)

        dps = []
        resume_file = 'O3_forecasts_%s.csv' % dt.strftime('%Y%m%d_%H%M%S')
        resume_file_path = '%s/%s' % (self.cfg['ftp']['localFolders']['results'], resume_file)

        # open the resume file and insert the metadata related to the predictions
        fw = open(resume_file_path, 'w')
        fw = self.create_metadata_in_file(fw, dt)

        # cycle all over the results files
        res_files_array = glob('%s/*.mat' % self.cfg['local']['outputMat'])
        res_files_array.sort()
        for results_file in res_files_array:

            tmp = results_file.split(os.path.sep)
            (location_code, case_code, predictor_code) = tmp[-1][:-4].split('_')
            str_vals = location_code

            fields = dict()
            data = sio.loadmat(file_name=results_file)

            # check if the output is the difference with the previous day or the actual value
            if 'use_diff' in data.keys() and data['use_diff'][0][0] == 1:
                offset = self.get_offset(location=location_code)
            else:
                offset = 0

            # get the probabilities
            for i in range(0, len(data['classProbability'])):
                class_probability = float(data['classProbability'][i][0])
                fields['prob_class_%d' % (i+1)] = round(class_probability, 1)
                str_vals = '%s,%.1f' % (str_vals, class_probability)
                
            # get the predicted value performed by the ensemble
            forecast_value = float(data['predictedValue'][0][0]) + offset
            fields['forecast'] = round(forecast_value, 1)
            str_vals = '%s,%.1f' % (str_vals, forecast_value)

            # get the predicted value performed by the RF
            forecast_value_rf = float(data['predictedValueRF'][0][0]) + offset
            fields['forecastRF'] = round(forecast_value_rf, 1)

            # get the percentiles (0, 10, 20, etc.)
            for i in range(0, 11):
                q = float(data['quantiles'][0][i*100]) + offset
                fields['perc_%03d' % (i*10)] = round(q, 1)
                str_vals = '%s,%.1f' % (str_vals, q)

            # Append a 'p' to stringify the tags different than latest
            if predictor_code != 'latest':
                predictor_code = 'p_%s' % predictor_code

            point = {
                        'time': self.predicted_day,
                        'measurement': self.cfg['influxDB']['measurementForecasts'],
                        'fields': fields,
                        'tags': dict(location=location_code, case=self.forecast_type, predictor=predictor_code)
                    }
            dps.append(point)

            # Insert the quantiles
            for i in range(0, len(data['quantiles'][0])):
                point = {
                            'time': self.predicted_day + (i*60),
                            'measurement': self.cfg['influxDB']['measurementForecastsQuantiles'],
                            'fields': dict(value=float(data['quantiles'][0][i]) + offset),
                            'tags': dict(location=location_code, case=self.forecast_type, predictor=predictor_code)
                        }
                dps.append(point)

            # Write the values string in the resume file
            if predictor_code == 'latest':
                fw.write('%s\n' % str_vals)

            # Insert the data in the DB if too many
            if len(dps) >= int(self.cfg['influxDB']['maxLinesPerInsert']):
                self.logger.info('Sent %i points to InfluxDB server' % len(dps))
                self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])
                dps = []

        fw.close()

        self.logger.info('Sent %i points to InfluxDB server' % len(dps))
        self.influxdb_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])

        self.logger.info('Data saved in file %s' % resume_file_path)

        # send the file to a remote FTP server
        if self.cfg['forecastPeriod']['case'] == 'current':
            self.logger.info('Send file \'%s\' to FTP server %s' % (resume_file_path, self.cfg['ftp']['host']))
            try:
                ftp = ftplib.FTP(self.cfg['ftp']['host'])
                ftp.login(self.cfg['ftp']['user'], self.cfg['ftp']['password'])
                ftp.cwd(self.cfg['ftp']['remoteFolders']['results'])
                with open(resume_file_path, 'rb') as f:
                    ftp.storbinary('STOR %s' % resume_file, f)
            except Exception as e:
                self.logger.error('Connection exception: %s' % str(e))
            ftp.close()

        # delete the local file
        os.unlink(resume_file_path)

    def create_metadata_in_file(self, fw, dt):
        """
        Create metadata section and header for the resume file
        """
        # write the metadata
        fw.write('Informazioni sulla previsione\n')
        if self.forecast_type == 'MOR':
            fw.write('Tipo: previsione della mattina del giorno D per la sera del giorno D\n')
        elif self.forecast_type == 'EVE':
            fw.write('Tipo: previsione della sera del giorno D-1 per la sera del giorno D\n')
        fw.write('Creazione: %s\n' % dt.strftime('%Y-%m-%d %H:%M:%S'))
        fw.write('\n')

        # write the header
        str_header = 'Località'
        for limit in constants.OZONE_INDEXES_LIMITS:
            str_header = '%s,%s' % (str_header, limit)
        str_header = '%s,Massimo giornaliero predetto [μg/m³]' % str_header
        for qi in range(0, 11):
            str_header = '%s,Percentile %i' % (str_header, qi*10)
        fw.write('%s\n' % str_header)

        return fw

    def clear(self):
        """
        Clear input/output folders
        """
        try:
            # clear input folder
            self.logger.info('Clear input folder %s' % self.cfg['local']['inputMat'])
            for file_to_delete in glob('%s/*.mat' % self.cfg['local']['inputMat']):
                os.unlink(file_to_delete)

            # clear output folder
            self.logger.info('Clear output folder %s' % self.cfg['local']['outputMat'])
            for file_to_delete in glob('%s/*.mat' % self.cfg['local']['outputMat']):
                os.unlink(file_to_delete)
        except Exception as e:
            self.logger.error('Unable to clear input/output folders')
            self.logger.error('Exception: %s' % str(e))

    def get_offset(self, location):
        dt = datetime.utcfromtimestamp(self.predicted_day)
        dt_day_before = dt - timedelta(days=1)

        query = 'SELECT mean(value) FROM %s WHERE location=\'%s\' AND ' \
                'signal=\'%s\' AND time>=\'%s\' AND time<=\'%s\' ' \
                'GROUP BY time(1h)' % (self.cfg['influxDB']['measurementOASI'], location, 'O3',
                                       '%sT00:00:00Z' % dt_day_before.strftime('%Y-%m-%d'),
                                       '%sT23:59:59Z' % dt_day_before.strftime('%Y-%m-%d'))

        self.logger.info('Performing query: %s' % query)
        res = self.influxdb_client.query(query, epoch='s')

        vals = []
        try:
            for i in range(0, len(res.raw['series'][0]['values'])):
                if res.raw['series'][0]['values'][i][1] is not None:

                    val = float(res.raw['series'][0]['values'][i][1])
                    vals.append(val)
        except Exception as e:
            self.logger.error('Unable to calculate the maximum of day before')
            return 0

        # return the maximum
        return np.max(vals)
