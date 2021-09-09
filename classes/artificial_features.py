# import section
import numpy as np
import pytz

from influxdb import InfluxDBClient
from datetime import date, datetime, timedelta


class ArtificialFeatures:
    """
    Class handling the forecasting of a couple location_case (e.g. BIO_MOR)
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

    def analyze_signal(self, signal):

        val = np.nan

        if len(signal.split('__')) == 1:
            if 'KLO' in signal:
                measurement = self.cfg['influxDB']['measurementMeteoSuisse']
                val = self.do_KLO_query(signal, measurement)

            if 'Totale' in signal:
                measurement = self.cfg['influxDB']['measurementGlobal']
                val = self.do_IFEC_query(signal, measurement)

        else:
            tmp = signal.split('__')

            if 'T_2M' in tmp[1] and '12h_mean' in tmp[2]:
                measurement = self.cfg['influxDB']['measurementMeteoSuisse']
                val = self.do_T_2M_query(signal, measurement)

            if 'MAX' in tmp[2]:
                measurement = self.cfg['influxDB']['measurementMeteoSuisse']
                val = self.do_MAX_query(signal, measurement)

            if 'transf' in tmp[2]:
                measurement = self.cfg['influxDB']['measurementMeteoSuisse']
                val = self.do_transf_query(signal, measurement)

            if 'NOx__12h' in signal or '24h' in tmp[2] or '48h' in tmp[2] or '72h' in tmp[2]:
                measurement = self.cfg['influxDB']['measurementOASI']
                val = self.do_multiday_query(signal, measurement)

            if 'mean_mor' in tmp[2] or 'mean_eve' in tmp[2]:
                measurement = self.cfg['influxDB']['measurementMeteoSuisse']
                val = self.do_mor_eve_query(signal, measurement)

            if 'TOT_PREC' in tmp[1]:
                measurement = self.cfg['influxDB']['measurementMeteoSuisse']
                val = self.do_tot_prec_query(signal, measurement)

        if val == np.nan:
            self.logger.error('Unrecognized artificial feature in function analyze_signal')

        return val

    def do_IFEC_query(self, signal_data, measurement):
        """Total_NOx is very easy, just query the already existing value in the DB; however, for Total_VOC it is
        necessary to query the existing value, which combine traffic (roads, highways, planes), combustion, agricolture,
        and industry, then add the daily calculated Woods emission with the below formula.
        Do note that the signal stored in the DB does NOT account for wood emission!"""

        dt = self.set_forecast_day()
        lcl_dt = dt.strftime('%Y-%m-%d')
        lcl_dt_plus_one_day = (dt + timedelta(days=1)).strftime('%Y-%m-%d')
        lcl_dt_MOR = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
        lcl_dt_EVE = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')
        func = 'mean'

        if 'NOx' in signal_data:
            if self.forecast_type == 'MOR':
                query_NOx = 'SELECT value FROM %s WHERE signal=\'%s\' AND time=\'%s\'' % (measurement, 'Total_NOx', lcl_dt)
            else:
                query_NOx = 'SELECT value FROM %s WHERE signal=\'%s\' AND time=\'%s\'' % (measurement, 'Total_NOx', lcl_dt_plus_one_day)

            self.logger.info('Performing query: %s' % query_NOx)
            NOx = self.influxdb_client.query(query_NOx, epoch='s').raw['series'][0]['values'][0][1]
            return NOx

        elif 'VOC' in signal_data:

            # Load necessary constants
            T_s = 303
            R = 8.314
            alpha = 0.0027
            C_L1 = 1.066
            C_T1 = 95000
            C_T2 = 230000
            T_m = 314
            C_T3 = 0.961

            # Get value without woods emission
            if self.forecast_type == 'MOR':
                query_VOC = 'SELECT value FROM %s WHERE signal=\'%s\' AND time=\'%s\'' % (measurement, 'Total_VOC', lcl_dt)
            else:
                query_VOC = 'SELECT value FROM %s WHERE signal=\'%s\' AND time=\'%s\'' % (measurement, 'Total_VOC', lcl_dt_plus_one_day)

            VOC = self.influxdb_client.query(query_VOC, epoch='s').raw['series'][0]['values'][0][1]

            measurement_MS = self.cfg['influxDB']['measurementMeteoSuisse']
            # Get the 24 hours mean of forecasted irradiance and temperature

            if self.forecast_type == 'MOR':
                steps_G = self.create_forecast_chunk_steps_string(1, 24)
                steps_T = self.create_forecast_chunk_steps_string(0, 23)
                query_LUG_G = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' % (measurement_MS, 'LUG', 'GLOB', steps_G, lcl_dt_MOR)
                query_LUG_T = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' % (measurement_MS, 'LUG', 'T_2M', steps_T, lcl_dt_MOR)
            else:
                steps_G = self.create_forecast_chunk_steps_string(12, 33)
                steps_T = self.create_forecast_chunk_steps_string(12, 33)
                query_LUG_G = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' % (measurement_MS, 'LUG', 'GLOB', steps_G, lcl_dt_EVE)
                query_LUG_T = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' % (measurement_MS, 'LUG', 'T_2M', steps_T, lcl_dt_EVE)
            self.logger.info('Performing query: %s' % query_LUG_G)
            Q = self.influxdb_client.query(query_LUG_G, epoch='s').raw['series'][0]['values'][0][1]
            self.logger.info('Performing query: %s' % query_LUG_T)
            T = self.influxdb_client.query(query_LUG_T, epoch='s').raw['series'][0]['values'][0][1]

            # Transform into the appropriate unit of measurement
            Q_ = Q * 4.6
            T_ = T + 273.14

            # Calculate woods emission
            gamma = (alpha*C_L1*Q_/np.sqrt(1+alpha*alpha*Q_*Q_)) * (np.exp(C_T1*(T_-T_s)/(R*T_s*T_))) / (1+np.exp(C_T2*(T_-T_m)/(R*T_s*T_)))
            emission = 98340 * gamma

            return VOC + emission

        else:
            self.logger.error('Something wrong in IFEc features calculation')
            return np.nan

        ### For testing

        # T_s = 303
        # R = 8.314
        # alpha = 0.0027
        # C_L1 = 1.066
        # C_T1 = 95000
        # C_T2 = 230000
        # T_m = 314
        # C_T3 = 0.961
        #
        # for d in range(1, 2):
        #     dt = self.set_forecast_day()
        #     dt = dt.replace(year=int(2018), month=int(6), day=int(d))
        #     dt_ = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
        #
        #     Q = 23.83333333333333 * 4.6
        #     T = 14.613194444444444 + 273.13
        #
        #     print(Q)
        #     print(T)
        #
        #     T_ = (R*T_s*T)
        #     # C_L = alpha * C_L1 * Q / np.sqrt(1 + alpha**2*Q**2)
        #     C_L = alpha * C_L1 * Q / np.sqrt(1 + (alpha*alpha*Q*Q))
        #     C_T = np.exp(C_T1 * (T - T_s) / T_) / (1 + np.exp(C_T2 * (T - T_m) / T_))
        #     gamma2 = C_L * C_T
        #
        #     gamma = (alpha*C_L1*Q/np.sqrt(1+alpha*alpha*Q*Q)) * (np.exp(C_T1*(T-T_s)/(R*T_s*T))) / (1+np.exp(C_T2*(T-T_m)/(R*T_s*T)))
        #     gamma3 = (alpha*C_L1*Q/np.sqrt(1+alpha*alpha*Q*Q)) * (np.exp(C_T1*(T-T_s)/T_)) / (1+np.exp(C_T2*(T-T_m)/T_))
        #     emission = 98340 * gamma
        #     print(dt_[:10] + ': ' + str(emission))
        #     print(dt_[:10] + ': ' + str(gamma))
        #     print(dt_[:10] + ': ' + str(gamma2))
        #     print(dt_[:10] + ': ' + str(gamma3))


    def do_mor_eve_query(self, signal_data, measurement):
        (location, signal_code, aggregator) = signal_data.split('__')
        dt = self.set_forecast_day()
        func = 'mean'

        if self.forecast_type == 'MOR':
            lcl_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
            if aggregator == 'mean_mor':
                if signal_code == 'GLOB':
                    steps = self.create_forecast_chunk_steps_string(1, 7)  # start steps at 1 instead of 0 because step00 is always -99999
                else:
                    steps = self.create_forecast_chunk_steps_string(0, 7)
            elif aggregator == 'mean_eve':
                steps = self.create_forecast_chunk_steps_string(7, 19)
        else:
            lcl_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')
            if aggregator == 'mean_mor':
                steps = self.create_forecast_chunk_steps_string(10, 22)
            elif aggregator == 'mean_eve':
                steps = self.create_forecast_chunk_steps_string(22, 23)

        query = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\''\
                % (measurement, location, signal_code, steps, lcl_dt)
        self.logger.info('Performing query: %s' % query)
        return self.calc_data(query=query, signal_data=signal_data, func=func)

    def do_MAX_query(self, signal_data, measurement):
        (location, signal_code, aggregator) = signal_data.split('__')
        dt = self.set_forecast_day()
        func = 'max'
        steps = self.create_forecast_chunk_steps_string(0, 33)

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            lcl_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
        else:
            lcl_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')

        query = 'SELECT max("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' \
                % (measurement, location, signal_code, steps, lcl_dt)
        self.logger.info('Performing query: %s' % query)
        return self.calc_data(query=query, signal_data=signal_data, func=func)

    def do_T_2M_query(self, signal_data, measurement):
        (location, signal_code, aggregator) = signal_data.split('__')
        dt = self.set_forecast_day()
        func = 'mean'

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            lcl_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
            steps = self.create_forecast_chunk_steps_string(9, 21)
        else:
            lcl_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')
            steps = self.create_forecast_chunk_steps_string(20, 33)

        query = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' \
                % (measurement, location, signal_code, steps, lcl_dt)
        self.logger.info('Performing query: %s' % query)

        val = self.calc_data(query=query, signal_data=signal_data, func=func) - 273.1

        if 'squared' in aggregator:
            return val**2
        else:
            return val

    def do_transf_query(self, signal_data, measurement):
        (location, signal_code, aggregator) = signal_data.split('__')
        dt = self.set_forecast_day()
        func = 'max'
        steps = self.create_forecast_chunk_steps_string(0, 33)

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            lcl_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
        else:
            lcl_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')

        query = 'SELECT max("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' \
                % (measurement, location, 'TD_2M', steps, lcl_dt)
        self.logger.info('Performing query: %s' % query)
        res = self.calc_data(query=query, signal_data=signal_data, func=func)
        return (res+20)**3

    def do_tot_prec_query(self, signal_data, measurement):
        (location, signal_code, aggregator) = signal_data.split('__')
        dt = self.set_forecast_day()
        func = 'mean'

        # start steps at 1 instead of 0 because step00 is always -99999
        steps = self.create_forecast_chunk_steps_string(1, 23)

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            lcl_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')

        else:
            lcl_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')

        query = 'SELECT sum("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND %s AND time=\'%s\'' \
                % (measurement, location, signal_code, steps, lcl_dt)
        self.logger.info('Performing query: %s' % query)
        return self.calc_data(query=query, signal_data=signal_data, func=func)

    def do_KLO_query(self, signal_data, measurement):
        """
        Calculate the pressure gradient between Kloten airport and Lugano to take into account the wind presence
        """

        dt = self.set_forecast_day()

        # the last forecast has been performed at 03:00 or at 12:00
        if self.forecast_type == 'MOR':
            lcl_dt = '%sT03:00:00Z' % dt.strftime('%Y-%m-%d')
        else:
            lcl_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')

        query_LUG = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time=\'%s\'' % (measurement, 'LUG', 'PMSL', lcl_dt)
        query_KLO = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time=\'%s\'' % (measurement, 'KLO', 'PMSL', lcl_dt)

        self.logger.info('Performing query: %s' % query_LUG)
        res_LUG = self.influxdb_client.query(query_LUG, epoch='s').raw['series'][0]['values'][0][1]
        self.logger.info('Performing query: %s' % query_KLO)
        res_KLO = self.influxdb_client.query(query_KLO, epoch='s').raw['series'][0]['values'][0][1]
        return (res_KLO - res_LUG)/100.0

    def do_multiday_query(self, signal_data, measurement):
        """
        Here we calculate the mean of the last 24, 48 or 72 hourly measurements
        Special case NOx_12h is also calculated here
        """

        (location, signal_code, aggregator) = signal_data.split('__', 2)
        dt = self.set_forecast_day()
        func = 'mean'

        if '12h' in aggregator:
            if self.forecast_type == 'MOR':
                start_dt = '%sT12:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                end_dt = '%sT00:00:00Z' % dt.strftime('%Y-%m-%d')
            else:
                start_dt = '%sT00:00:00Z' % dt.strftime('%Y-%m-%d')
                end_dt = '%sT12:00:00Z' % dt.strftime('%Y-%m-%d')

        if '24h' in aggregator:
            if self.forecast_type == 'MOR':
                start_dt = '%sT05:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                end_dt = '%sT04:00:00Z' % dt.strftime('%Y-%m-%d')
            else:
                start_dt = '%sT17:00:00Z' % (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                end_dt = '%sT16:00:00Z' % dt.strftime('%Y-%m-%d')

        elif '48h' in aggregator:
            if self.forecast_type == 'MOR':
                start_dt = '%sT05:00:00Z' % (dt - timedelta(days=2)).strftime('%Y-%m-%d')
                end_dt = '%sT04:00:00Z' % dt.strftime('%Y-%m-%d')
            else:
                start_dt = '%sT17:00:00Z' % (dt - timedelta(days=2)).strftime('%Y-%m-%d')
                end_dt = '%sT16:00:00Z' % dt.strftime('%Y-%m-%d')

        elif '72h' in aggregator:
            if self.forecast_type == 'MOR':
                start_dt = '%sT05:00:00Z' % (dt - timedelta(days=3)).strftime('%Y-%m-%d')
                end_dt = '%sT04:00:00Z' % dt.strftime('%Y-%m-%d')
            else:
                start_dt = '%sT17:00:00Z' % (dt - timedelta(days=3)).strftime('%Y-%m-%d')
                end_dt = '%sT16:00:00Z' % dt.strftime('%Y-%m-%d')

        else:
            self.logger.error('Unexpected error with 12h, 24h, 48h, 72h artificial signal')

        # Measured data from MeteoSwiss station in Lugano are in InfluxDB under measurement=meteosuisse_data, so
        # applying a patch here (DM 20.07.21)
        if location == 'MS-LUG':
            measurement = self.cfg['influxDB']['measurementMeteoSuisse']

        query = 'SELECT mean("value") FROM %s WHERE location=\'%s\' AND signal=\'%s\' AND time>=\'%s\' AND ' \
                'time<=\'%s\'' % (measurement, location, signal_code, start_dt, end_dt)
        self.logger.info('Performing query: %s' % query)

        return self.calc_data(query=query, signal_data=signal_data, func=func)

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

