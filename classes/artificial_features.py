# import section
from datetime import datetime, timedelta

import numpy as np
import pytz
from influxdb import InfluxDBClient


class ArtificialFeatures:
    """Class handling the forecasting of features calculated from the other measurements or forecasts (e.g. VOC,
    Kloten-Luano pressure gradient, specific time slots means, and so on...)
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
        Perform a query on signals who are location independent, i.e. it doesnt matter where they are measured.
        This method is used for calculating VOC and NOx daily variables in method do_IFEC_query

        :param signal: signal code
        :type signal: str
        :return: single value of the queried signal
        :rtype: float
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
        function, such as mean, max, min or sum. This method is used in most methods except KLO-LUG and past
        measurements features

        :param measurement: InfluxDB measurement code
        :type measurement: str
        :param signal_data: signal code
        :type signal_data: str
        :param steps: steps (hours) in the future for the data we're interested in
        :type steps: str
        :param func: reduction to apply to multiple data ('min', 'max' or 'mean')
        :type func: str
        :return: single value of the queried signal
        :rtype: float
        """
        (location, signal_code, aggregator) = signal_data.split('__')
        dt = self.set_forecast_day()

        if self.forecast_type == 'MOR':
            lcl_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
        else:
            lcl_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')

        query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' \
                % (measurement, location, signal_code, steps, lcl_dt)
        return self.calc_data(query=query, signal_data=signal_data, func=func)

    def get_query_value_measure(self, measurement, signal_data, start_dt, end_dt, func):
        """
        Perform a query on measured signals using a starting and ending time and a data aggregating function

        :param measurement: InfluxDB measurement code
        :type measurement: str
        :param signal_data: signal code
        :type signal_data: str
        :param start_dt: timestamp, format '%Y-%m-%dTHH:MM:SSZ'
        :type start_dt: str
        :param end_dt: timestamp, format '%Y-%m-%dTHH:MM:SSZ'
        :type end_dt: str
        :param func: reduction to apply to multiple data ('min', 'max' or 'mean')
        :type func: str
        :return: single value of the queried signal
        :rtype: float
        """
        (location, signal_code, aggregator) = signal_data.split('__', 2)

        query = 'SELECT value FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time>=\'%s\' AND ' \
                'time<=\'%s\'' % (measurement, location, signal_code, start_dt, end_dt)
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

        :param days: number of days back in time
        :type days: int
        :return: list of timestamps
        :rtype: list
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

        :param signal: signal code
        :type signal: str
        :return: single value of the queried signal
        :rtype: float
        """

        val = np.nan

        if len(signal.split('__')) == 1:
            if 'KLO' in signal:
                # KLO-LUG, KLO-LUG_favonio
                val = self.do_KLO_query(signal)

            if signal == 'NOx_Totale':
                val = self.get_query_value_global('Total_NOx')

            # This signal is not handled as global anymore
            # if signal == 'VOC_Totale':
            #     VOC_without_woods, VOC_woods, VOC_woods_corrected = self.do_VOC_query()
            #     if self.cfg['VOC']['useCorrection'] and self.cfg['VOC'][
            #         'emissionType'] == 'forecasted' and self.VOC_forecasted_status:
            #         self.logger.info('Correcting forecasted VOC woods emissions')
            #         val = VOC_without_woods + VOC_woods_corrected
            #     else:
            #         val = VOC_without_woods + VOC_woods

        else:
            tmp = signal.split('__')

            # if 'T_2M' in tmp[1] and '12h_mean' in tmp[2]:
            #     # E.g. P_BIO__T_2M__12h_mean, P_BIO__T_2M__12h_mean_squared
            #     measurement = self.cfg['influxDB']['measurementInputsForecasts']
            #     val = self.do_T_2M_query(signal, measurement)

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

    def calculate_wood_emission(self, region):
        """
        Calculate the VOC emissions from woods, using either forecasted or measured data. Forecasted data are not always
        available, so sometimes we're forced to use measured values and infer the future.

        :param emissionType: type of emission, either 'measured' or 'forecasted'
        :type emissionType: str
        :return: calculated VOC emission value
        :rtype: float
        """

        # Load necessary constants
        temp_s = self.cfg['VOC']['T_s']
        r = self.cfg['VOC']['R']
        alpha = self.cfg['VOC']['alpha']
        c_l1 = self.cfg['VOC']['C_L1']
        c_temp1 = self.cfg['VOC']['C_T1']
        c_temp2 = self.cfg['VOC']['C_T2']
        T_m = self.cfg['VOC']['T_m']
        # C_T3 = self.cfg['VOC']['C_T3']

        q, temp = self.get_Q_T_forecasted(region)

        # if emissionType == 'forecasted':
        #
        # elif emissionType == 'measured':
        #     [Q, T] = self.get_Q_T_measured()
        # else:
        #     self.logger.error('Unrecognized VOC woods type of data to use')
        #     [Q, T] = [None, None]

        # Calculate woods emission
        gamma = (alpha * c_l1 * q / np.sqrt(1 + np.power(alpha, 2) * np.power(q, 2))) * (np.exp(c_temp1 * (temp - temp_s) / (r * temp_s * temp))) / (1 + np.exp(c_temp2 * (temp - T_m) / (r * temp_s * temp)))
        return self.cfg['VOC']['KG_per_gamma'] * gamma

    def get_Q_T_forecasted(self, region):

        # Get 24 hours of forecasts
        if self.forecast_type == 'MOR':
            steps_G = self.create_forecast_chunk_steps_string(1, 24)
            steps_T = self.create_forecast_chunk_steps_string(0, 23)
        else:
            steps_G = steps_T = self.create_forecast_chunk_steps_string(10, 33)

        q = self.get_query_value_forecast(self.cfg['influxDB']['measurementInputsForecasts'],
                                          '%s__GLOB__' % self.cfg['regions'][region]['VOCStation'], steps_G, 'mean')
        temp = self.get_query_value_forecast(self.cfg['influxDB']['measurementInputsForecasts'],
                                             '%s__T_2M__' % self.cfg['regions'][region]['VOCStation'], steps_T, 'mean')

        # Transform into the appropriate unit of measurement
        q = q * self.cfg['VOC']['GLOB_to_PAR']

        return q, temp

    # def get_Q_T_measured(self):
    #
    #     self.VOC_forecasted_status = False
    #
    #     dt = self.set_forecast_day()
    #     func = 'mean'
    #     measurement = self.cfg['influxDB']['measurementInputsMeasurements']
    #     location = 'MS-LUG'
    #
    #     start_dt = '%sT23:05:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
    #     end_dt = '%sT23:05:00Z' % dt.strftime('%Y-%m-%d')
    #
    #     Q = self.get_query_value_measure(measurement, location + '__Gl__', start_dt, end_dt, func)
    #     T = self.get_query_value_measure(measurement, location + '__T__', start_dt, end_dt, func)
    #
    #     # Transform into the appropriate unit of measurement
    #     Q_ = Q * self.cfg['VOC']['GLOB_to_PAR']
    #     T_ = T + 273.1
    #
    #     return [Q_, T_]

    # This functions should not be used anymore
    def do_VOC_query(self, region):
        """
        Get values with and without woods emission
        """
        voc_other = self.get_query_value_global('Total_VOC')
        voc_woods_raw, voc_woods_corrected = self.calc_voc_woods(region)

        return voc_other, voc_woods_raw, voc_woods_corrected

    def calc_voc_woods(self, region):
        """
        Get values with and without woods emission
        """
        voc_woods_raw = self.calculate_wood_emission(region)
        voc_woods_corrected = self.cfg['VOC']['correction']['slope'] * voc_woods_raw + self.cfg['VOC']['correction']['intercept']
        return voc_woods_raw, voc_woods_corrected

    def do_mor_eve_query(self, signal_data, measurement):
        # mean_mor: Mean of values from 03:00 UTC to 10:00 UTC
        # mean_eve: Mean of values from 11:00 UTC to 21:00 UTC
        # Forecasted signals considered: GLOB, CLCT

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
        Calculate maximum value of all hourly forecasted temperatures

        :param signal_data: signal code
        :type signal_data: str
        :param measurement: InfluxDB measurement code
        :type measurement: str
        :return: single value of the queried signal
        :rtype: float
        """

        func = 'max'
        steps = self.create_forecast_chunk_steps_string(0, 40)

        return self.get_query_value_forecast(measurement, signal_data, steps, func)

    def do_T_2M_query(self, signal_data, measurement):
        # MOR: mean values of hourly forecasted temperatures from 12:00 to 00:00 of the same day
        # EVE: mean values of hourly forecasted temperatures from 10:00 to 21:00 of the following day
        # The squared value of the above means is a considered signal too

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

    def do_KLO_query(self, signal_data):
        """
        Calculate the pressure gradient between Kloten airport and Lugano to take into account the air current between
        South and North of Switzerland

        :param signal_data: signal code
        :type signal_data: str
        :return: single value of the queried signal
        :rtype: float
        """

        func = 'mean'
        dt = self.set_forecast_day()

        if dt < pytz.timezone("UTC").localize(datetime(2021, 4, 1)):
            measurement = self.cfg['influxDB']['measurementInputsMeasurements']
            start_dt = '%sT23:05:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
            end_dt = '%sT23:05:00Z' % dt.strftime('%Y-%m-%d')
            res_LUG = self.get_query_value_measure(measurement, 'LUG__P_red__', start_dt, end_dt, func)
            res_KLO = self.get_query_value_measure(measurement, 'KLO__P_red__', start_dt, end_dt, func)
            diff = res_KLO - res_LUG

        else:
            measurement = self.cfg['influxDB']['measurementInputsForecasts']
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

        :param signal_data: signal code
        :type signal_data: str
        :param measurement: InfluxDB measurement code
        :type measurement: str
        :return: single value of the queried signal
        :rtype: float
        """

        (location, signal_code, aggregator) = signal_data.split('__', 2)
        dt = self.set_forecast_day()
        func = signal_data.split('__')[-1]

        if '12h' in aggregator:
            if self.forecast_type == 'MOR':
                start_dt = '%sT10:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                end_dt = '%sT22:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                start_dt = '%sT22:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                end_dt = '%sT10:00:00Z' % dt.strftime('%Y-%m-%d')

        elif '24h' in aggregator:
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
        # tmp = signal_data.split('__')
        try:
            for i in range(0, len(res.raw['series'][0]['values'])):
                if res.raw['series'][0]['values'][i][1] is not None:

                    # The training of Meteosuisse temperatures were done in Kelvin degrees
                    # if tmp[1] in ['TD_2M', 'T_2M'] and 'MAX' not in signal_data and 'transf' not in signal_data:
                    #     val = float(res.raw['series'][0]['values'][i][1]) + 273.1
                    # else:
                    #     val = float(res.raw['series'][0]['values'][i][1])
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
            elif func == 'std':
                return np.std(vals)
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
