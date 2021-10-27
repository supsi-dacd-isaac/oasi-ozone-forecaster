# import section
from datetime import datetime, timedelta

import numpy as np
import pytz
from influxdb import InfluxDBClient


class ArtificialFeatures:
    """
    Class handling the forecasting of features calculated from the other measurements or forecasts (e.g. VOC,
    Kloten-Luano pressure gradient, ...)
    """

    def __init__(self, influxdb_client, forecast_type, cfg, logger):
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
        """
        # set the variables
        self.influxdb_client = influxdb_client
        self.forecast_type = forecast_type
        self.cfg = cfg
        self.logger = logger
        self.input_data = dict()
        self.input_data_desc = []
        self.input_data_values = []
        self.day_to_predict = None
        self.cfg_signals = None
        self.input_df = None

    def get_query_value_global(self, signal):
        """
        Perform a query on signals independent from the location where they are measured. This method is used for
        calculating VOC and NOx variables in method do_IFEC_query
        """

        measurement = self.cfg["influxDB"]["measurementGlobal"]
        dt = self.set_forecast_day()
        lcl_dt = dt.strftime('%Y-%m-%d')
        lcl_dt_plus_one_day = (dt + timedelta(days=1)).strftime('%Y-%m-%d')

        if self.forecast_type == 'MOR':
            query = 'SELECT value FROM %s WHERE signal=\'%s\' AND time=\'%s\'' % (measurement, signal, lcl_dt)
        else:
            query = 'SELECT value FROM %s WHERE signal=\'%s\' AND time=\'%s\'' % (
                measurement, signal, lcl_dt_plus_one_day)

        self.logger.info('Performing query: %s' % query)
        value = self.influxdb_client.query(query, epoch='s').raw['series'][0]['values'][0][1]

        return value

    def get_query_value_forecast(self, measurement, signal_data, steps, func):
        """
        Perform a query on forecasted signals using a certain amount of steps forward in time and a data aggregating
        function, such as mean, max, min or sum.
        This method is used in most methods except KLO-LUG and past measurements features
        """

        (location, signal_code, aggregator) = signal_data.split('__')
        dt = self.set_forecast_day()

        if self.forecast_type == 'MOR':
            lcl_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
        else:
            lcl_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')

        query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' \
                % (measurement, location, signal_code, steps, lcl_dt)
        self.logger.info('Performing query: %s' % query)
        return self.calc_data(query=query, signal_data=signal_data, func=func)

    def get_query_value_measure(self, measurement, signal_data, start_dt, end_dt, func):

        (location, signal_code, aggregator) = signal_data.split('__', 2)

        query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time>=\'%s\' AND ' \
                'time<=\'%s\'' % (measurement, location, signal_code, start_dt, end_dt)
        self.logger.info('Performing query: %s' % query)

        return self.calc_data(query=query, signal_data=signal_data, func=func)

    def steps_type_forecast(self, mor_start, mor_end, eve_start, eve_end):
        """
        Create a string for the forward steps in time of a forecast signal query
        """

        if self.forecast_type == 'MOR':
            steps = self.create_forecast_chunk_steps_string(mor_start, mor_end)
        else:
            steps = self.create_forecast_chunk_steps_string(eve_start, eve_end)

        return steps

    def measurements_start_end(self, days):
        """
        Create start and end times for the measurement signal queries. To be used in the do_multiday_query method
        """

        dt = self.set_forecast_day()

        if self.forecast_type == 'MOR':
            start_dt = '%sT05:00:00Z' % (dt - timedelta(days=days)).strftime('%Y-%m-%d')
            end_dt = '%sT04:00:00Z' % dt.strftime('%Y-%m-%d')
        else:
            start_dt = '%sT17:00:00Z' % (dt - timedelta(days=days)).strftime('%Y-%m-%d')
            end_dt = '%sT16:00:00Z' % dt.strftime('%Y-%m-%d')

        return [start_dt, end_dt]

    def analyze_signal(self, signal):
        """
        Parse the signal codes and send it to the appropriate method for value calculation
        """

        val = np.nan

        if len(signal.split('__')) == 1:
            if 'KLO' in signal:
                # KLO-LUG, KLO-LUG_favonio
                measurement = self.cfg['influxDB']['measurementInputsForecasts']
                val = self.do_KLO_query(signal, measurement)

            if signal == 'NOx_Totale':
                val = self.get_query_value_global('Total_NOx')

            if signal == 'VOC_Totale':
                measurement = self.cfg['influxDB']['measurementInputsMeasurements']
                VOC_without_woods, VOC_woods, VOC_woods_corrected = self.do_VOC_query(measurement)
                if self.cfg['VOC']['useCorrection'] and self.cfg['VOC']['emissionType'] == 'forecasted':
                    val = VOC_without_woods + VOC_woods_corrected
                else:
                    val = VOC_without_woods + VOC_woods

        else:
            tmp = signal.split('__')

            if 'T_2M' in tmp[1] and '12h_mean' in tmp[2]:
                # E.g. P_BIO__T_2M__12h_mean, P_BIO__T_2M__12h_mean_squared
                measurement = self.cfg['influxDB']['measurementInputsForecasts']
                val = self.do_T_2M_query(signal, measurement)

            if 'MAX' in tmp[2]:
                # E.g. P_BIO__T_2M__MAX
                measurement = self.cfg['influxDB']['measurementInputsForecasts']
                val = self.do_MAX_query(signal, measurement)

            if 'transf' in tmp[2]:
                # E.g. P_BIO__TD_2M__transf
                measurement = self.cfg['influxDB']['measurementInputsForecasts']
                val = self.do_transf_query(signal, measurement)

            if 'NOx__12h' in signal or '24h' in tmp[2] or '48h' in tmp[2] or '72h' in tmp[2]:
                # E.g. BIO__CN__24h__mean, BIO__CN__48h__mean, BIO__CN__72h__mean
                measurement = self.cfg['influxDB']['measurementInputsMeasurements']
                val = self.do_multiday_query(signal, measurement)

            if 'mean_mor' in tmp[2] or 'mean_eve' in tmp[2]:
                # E.g. P_BIO__GLOB__mean_mor,P_BIO__GLOB__mean_eve, P_BIO__CLCT__mean_mor, P_BIO__CLCT__mean_eve
                measurement = self.cfg['influxDB']['measurementInputsForecasts']
                val = self.do_mor_eve_query(signal, measurement)

            if 'TOT_PREC' in tmp[1]:
                # E.g. P_BIO__TOT_PREC__sum
                measurement = self.cfg['influxDB']['measurementInputsForecasts']
                val = self.do_tot_prec_query(signal, measurement)

        if val == np.nan:
            self.logger.error('Unrecognized artificial feature in function analyze_signal')

        return val

    def calculate_wood_emission(self, emissionType):
        """
        Calculate the VOC emissions from woods, using either forecasted or measured data
        """

        # Load necessary constants
        T_s = self.cfg['VOC']['T_s']
        R = self.cfg['VOC']['R']
        alpha = self.cfg['VOC']['alpha']
        C_L1 = self.cfg['VOC']['C_L1']
        C_T1 = self.cfg['VOC']['C_T1']
        C_T2 = self.cfg['VOC']['C_T2']
        T_m = self.cfg['VOC']['T_m']
        # C_T3 = self.cfg['VOC']['C_T3']

        if emissionType == 'forecasted':
            [Q, T] = self.get_Q_T_forecasted()
        elif emissionType == 'measured':
            [Q, T] = self.get_Q_T_measured()
        else:
            self.logger.error('Unrecognized VOC woods type of data to use')

        # Calculate woods emission
        gamma = (alpha * C_L1 * Q / np.sqrt(1 + alpha * alpha * Q * Q)) * (
            np.exp(C_T1 * (T - T_s) / (R * T_s * T))) / (1 + np.exp(C_T2 * (T - T_m) / (R * T_s * T)))
        emission = self.cfg['VOC']['KG_per_gamma'] * gamma
        return emission

    def get_Q_T_forecasted(self):

        measurement_MS = self.cfg['influxDB']['measurementInputsForecasts']
        func = 'mean'

        # Get 24 hours of forecasts
        if self.forecast_type == 'MOR':
            steps_G = self.create_forecast_chunk_steps_string(1, 24)
            steps_T = self.create_forecast_chunk_steps_string(0, 23)
        else:
            steps_G = steps_T = self.create_forecast_chunk_steps_string(10, 33)

        Q = self.get_query_value_forecast(measurement_MS, 'LUG__GLOB__', steps_G, func)
        T_ = self.get_query_value_forecast(measurement_MS, 'LUG__T_2M__', steps_T, func)

        # Transform into the appropriate unit of measurement
        if Q is not None:
            Q_ = Q * self.cfg['VOC']['GLOB_to_PAR']
        else:
            # This should not happen, but if it happens we try to save the day and use measurements instead. Pray we
            # never enter here
            self.logger.error(
                'Warning: Forecasted data for calculating VOC not found. Trying again with measured data.')
            Q_ = self.get_Q_T_measured()[0]

        if T_ is None:
            self.logger.error(
                'Warning: Forecasted data for calculating VOC not found. Trying again with measured data.')
            T_ = self.get_Q_T_measured()[1]

        return [Q_, T_]

    def get_Q_T_measured(self):

        dt = self.set_forecast_day()
        func = 'mean'
        measurement = self.cfg['influxDB']['measurementInputsMeasurements']
        location = 'MS-LUG'

        start_dt = '%sT23:05:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
        end_dt = '%sT23:05:00Z' % dt.strftime('%Y-%m-%d')

        Q = self.get_query_value_measure(measurement, location + '__Gl__', start_dt, end_dt, func)
        T = self.get_query_value_measure(measurement, location + '__T__', start_dt, end_dt, func)

        # Transform into the appropriate unit of measurement
        Q_ = Q * self.cfg['VOC']['GLOB_to_PAR']
        T_ = T + 273.1

        return [Q_, T_]

    def do_VOC_query(self, measurement):
        """
        Total_NOx is easy, just query the previously calculated value in the DB. However, for Total_VOC it is necessary
        to query the existing value, which combines traffic (roads, highways, planes), combustion, agricolture,
        and industry, then add the daily calculated woods emission with the below iso gamma formula.
        Note that the VOC signal stored in the DB does NOT account for wood emission!
        """

        # Get values with and without woods emission
        VOC_without_woods = self.get_query_value_global('Total_VOC')
        VOC_woods = self.calculate_wood_emission(self.cfg['VOC']['emissionType'])

        VOC_woods_corrected = self.cfg['VOC']['correction']['slope'] * VOC_woods + self.cfg['VOC']['correction'][
            'intercept']

        return VOC_without_woods, VOC_woods, VOC_woods_corrected

    def do_mor_eve_query(self, signal_data, measurement):
        """
        mean_mor: Mean of values from 03:00 UTC to 10:00 UTC
        mean_eve: Mean of values from 11:00 UTC to 21:00 UTC
        Forecasted signals considered: GLOB, CLCT
        """

        (location, signal_code, aggregator) = signal_data.split('__')
        func = 'mean'

        if aggregator == 'mean_mor':
            if signal_code == 'GLOB':
                # Start at step01 because step00 is always -99999
                steps = self.steps_type_forecast(mor_start=1, mor_end=7, eve_start=15, eve_end=22)
            else:
                steps = self.steps_type_forecast(mor_start=0, mor_end=7, eve_start=15, eve_end=22)
        elif aggregator == 'mean_eve':
            steps = self.steps_type_forecast(mor_start=8, mor_end=18, eve_start=23, eve_end=33)
        else:
            self.logger.error('Something wrong in mean_mor, mean_eve features calculation')

        return self.get_query_value_forecast(measurement, signal_data, steps, func)

    def do_MAX_query(self, signal_data, measurement):
        """
        Maximum value of all hourly forecasted temperatures
        """

        func = 'max'
        steps = self.create_forecast_chunk_steps_string(0, 40)

        return self.get_query_value_forecast(measurement, signal_data, steps, func)

    def do_T_2M_query(self, signal_data, measurement):
        """
        MOR: mean values of hourly forecasted temperatures from 12:00 to 00:00 of the same day
        EVE: mean values of hourly forecasted temperatures from 10:00 to 21:00 of the following day
        The squared value of the above means is a considered signal too
        """

        func = 'mean'
        steps = self.steps_type_forecast(mor_start=7, mor_end=19, eve_start=22, eve_end=33)
        val = self.get_query_value_forecast(measurement, signal_data, steps, func) - 273.1

        if 'squared' in signal_data:
            return val ** 2
        else:
            return val

    def do_transf_query(self, signal_data, measurement):
        """
        Maximum value of all hourly forecasted Dew Temperatures, to which we add 20 and cube the obtained value
        """

        func = 'max'
        steps = self.create_forecast_chunk_steps_string(0, 40)
        res = self.get_query_value_forecast(measurement, signal_data, steps, func)

        return (res + 20) ** 3

    def do_tot_prec_query(self, signal_data, measurement):
        """
        Sum of the hourly forecasted precipitations for the next 24 hours
        """

        func = 'sum'
        # start steps at 1 instead of 0 because step00 is always -99999
        steps = self.create_forecast_chunk_steps_string(1, 23)

        return self.get_query_value_forecast(measurement, signal_data, steps, func)

    def do_KLO_query(self, signal_data, measurement):
        """
        Calculate the pressure gradient between Kloten airport and Lugano to take into account the air current between
        South and North of Switzerland
        """

        func = 'mean'
        steps = self.create_forecast_chunk_steps_string(0, 40)
        res_LUG = self.get_query_value_forecast(measurement, 'LUG__PMSL__0', steps, func)
        res_KLO = self.get_query_value_forecast(measurement, 'KLO__PMSL__0', steps, func)

        diff = (res_KLO - res_LUG) / 100.0

        if 'favonio' in signal_data:
            return 1.0 if diff >= 6.0 else 0.0
        else:
            return diff

    def do_multiday_query(self, signal_data, measurement):
        """
        Calculate the mean of the last 24, 48 or 72 hourly measurements for all measured signals
        Special case NOx_12h is also calculated here:
        MOR: mean value of NOx hourly measurements from 22:00 to 10:00 of previous day (previous afternoon)
        EVE: mean value of NOx hourly measurements from 22:00 of previous day to 10:00 of present day (present morning)
        """

        (location, signal_code, aggregator) = signal_data.split('__', 2)
        dt = self.set_forecast_day()
        func = 'mean'

        if '12h' in aggregator:
            if self.forecast_type == 'MOR':
                start_dt = '%sT10:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                end_dt = '%sT22:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                start_dt = '%sT22:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                end_dt = '%sT10:00:00Z' % dt.strftime('%Y-%m-%d')

        if '24h' in aggregator:
            [start_dt, end_dt] = self.measurements_start_end(days=1)

        elif '48h' in aggregator:
            [start_dt, end_dt] = self.measurements_start_end(days=2)

        elif '72h' in aggregator:
            [start_dt, end_dt] = self.measurements_start_end(days=3)

        else:
            self.logger.error('Unexpected error with 12h, 24h, 48h, 72h artificial signal')

        val = self.get_query_value_measure(measurement, location + '__' + signal_code + '__', start_dt, end_dt, func)

        return val

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
                    if tmp[1] in ['TD_2M', 'T_2M'] and 'MAX' not in signal_data and 'transf' not in signal_data:
                        val = float(res.raw['series'][0]['values'][i][1]) + 273.1
                    else:
                        val = float(res.raw['series'][0]['values'][i][1])
                    vals.append(val)

            if func == 'min':
                return np.min(vals)
            elif func == 'max':
                return np.max(vals)
            elif func == 'mean':
                return np.mean(vals)
            elif func == 'sum':
                return np.sum(vals)
        except Exception as e:
            self.logger.error('Forecast not available')
            self.logger.error('No data from query %s' % query)
            self.input_data[signal_data] = np.nan

    @staticmethod
    def create_forecast_chunk_steps_string(start, end):

        str_steps = '('
        for i in range(start, end + 1):
            str_steps += 'step=\'step%02d\' OR ' % i

        str_steps = '%s)' % str_steps[:-4]
        return str_steps
