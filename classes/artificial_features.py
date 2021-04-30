# import section
import json
import scipy.io as sio
import numpy as np
import pandas as pd
import pytz
import os


from influxdb import InfluxDBClient
from datetime import date, datetime, timedelta

import constants


class ArtificialFeatures:
    """
    Class handling the forecasting of a couple location_case (e.g. BIO_MOR)
    """

    def __init__(self, influxdb_client, forecast_type, location, cfg, logger):
        """
        Constructor
        :param influxdb_client: InfluxDB client
        :type influxdb_client: InfluxDBClient
        :param forecast_type: Forecast type (MOR | EVE)
        :type forecast_type: str
        :param location: Location
        :type location: str
        :param cfg: FTP parameters for the files exchange
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        """
        # set the variables
        self.influxdb_client = influxdb_client
        self.location = location
        self.forecast_type = forecast_type
        self.cfg = cfg
        self.logger = logger
        self.input_data = dict()
        self.input_data_desc = []
        self.input_data_values = []
        self.day_to_predict = None
        self.cfg_signals = None
        self.input_df = None

    def analyze_signal(self, signal):
        tmp = signal.split('__')

        if 'MAX' in tmp[2]:
            measurement = self.cfg['influxDB']['measurementMeteoSuisse']
            self.do_MAX_query(signal, measurement)

        if '24h' in tmp[2] or '48h' in tmp[2] or '72h' in tmp[2]:
            measurement = self.cfg['influxDB']['measurementOASI']
            self.do_multiday_query(signal, measurement)


    def do_MAX_query(self, signal_data, measurement):
        (location, signal_code, aggregator) = signal_data.split('__')
        dt = self.set_forecast_day()
        func = 'mean'

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            start_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
            end_dt = '%sT03:00:00Z' % (dt + timedelta(days=1)).strftime('%Y-%m-%d')

        else:
            start_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')
            end_dt = '%sT12:00:00Z' % (dt + timedelta(days=1)).strftime('%Y-%m-%d')

        query = 'SELECT max("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time>=\'%s\' AND ' \
                'time<=\'%s\'' % (measurement, location, signal_code, start_dt, end_dt)
        self.calc_data(query=query, signal_data=signal_data, func=func)

    def do_multiday_query(self, signal_data, measurement):
        pass

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

