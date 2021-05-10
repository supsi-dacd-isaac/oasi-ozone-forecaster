# import section
import json
import os
import pandas as pd
import numpy as np
import math
import pickle
from influxdb import InfluxDBClient

class Forecaster:
    """
    Class handling the forecasting of a couple location_case (e.g. BIO_MOR)
    """

    def __init__(self, influxdb_client, forecast_type, location, model_name, cfg, logger):
        """
        Constructor
        :param influxdb_client: InfluxDB client
        :type influxdb_client: InfluxDBClient
        :param forecast_type: Forecast type (MOR | EVE)
        :type forecast_type: str
        :param location: Location
        :type location: str
        :param model_name: Name of the model
        :type model_name: str
        :param cfg: FTP parameters for the files exchange
        :type cfg: dict
        :param logger: Logger
        :type logger: Logger object
        """
        # set the variables
        self.influxdb_client = influxdb_client
        self.forecast_type = forecast_type
        self.location = location
        self.model_name = model_name
        self.cfg = cfg
        self.logger = logger
        self.day_to_predict = None
        self.cfg_signals = None
        self.input_df = None
        self.available_features = 0
        self.unavailable_features = []
        self.unsurrogable_features = []
        self.predicted_value = 0
        self.perc_available_features = 0
        self.do_prediction = True
        self.prob_over_limit = 0

    def build_model_input_dataset(self, inputs_gatherer, input_cfg_file):
        """
        Build the dataset
        """
        self.day_to_predict = inputs_gatherer.day_to_predict
        self.cfg_signals = json.loads(open(input_cfg_file).read())

        input_data_values = []
        for signal in self.cfg_signals['signals']:
            # Take into account only the inputs needed by the model
            if signal in self.cfg_signals['signals']:
                input_data_values.append(inputs_gatherer.input_data[signal])

        # self.logger.info('Create the input dataframe')
        self.input_df = pd.DataFrame([input_data_values], columns=self.cfg_signals['signals'],
                                     index=[pd.DatetimeIndex([self.day_to_predict*1e9])])

        # Check inputs effective availability
        self.check_inputs_availability(inputs_gatherer.input_data_availability)

    def check_inputs_availability(self, inputs_availability):
        self.available_features = 0
        self.unavailable_features = []
        self.unsurrogable_features = []
        self.do_prediction = True
        for col in self.input_df.columns:
            if inputs_availability[col] is False:
                if math.isnan(self.input_df[col].values[0]):
                    self.logger.error('No surrogated data available for %s' % col)
                    self.do_prediction = False
                    self.unsurrogable_features.append(col)
                else:
                    self.logger.error('Data for code %s not available, used past values mean = %.1f' % (col, self.input_df[col].values[0]))
                self.unavailable_features.append(col)
            else:
                self.available_features += 1

    def binary_search(self, qrf, low, high, threshold):
        if high >= low:

            mid = (high + low) // 2

            # Get the middle and previous prediction values
            p_mid = qrf.predict(self.input_df, quantile=mid)
            p_prev = qrf.predict(self.input_df, quantile=mid-1)

            # Check if the solution has been reached
            if p_prev < threshold < p_mid:
                return mid, p_mid

            # If p_mid is greater than threshold, then it can only be present in left subarray
            elif p_mid > threshold:
                return self.binary_search(qrf, low, mid - 1, threshold)

            # Else (p_mid is smaller than threshold) the element can only be present in right subarray
            else:
                return self.binary_search(qrf, mid + 1, high, threshold)
        else:
            # Element is not present in the array
            return -1, -1

    def prob_overlimit(self, qrf):
        qrf1_percentiles = [qrf.predict(self.input_df, quantile=1.0), qrf.predict(self.input_df, quantile=99.0)]
        if max(qrf1_percentiles) < self.cfg['predictionSettings']['threshold']:
            return 0.0
        elif min(qrf1_percentiles) > self.cfg['predictionSettings']['threshold']:
            return 100.0
        else:
            # Binary search
            idx, _ = self.binary_search(qrf, 1.0, 99.0, self.cfg['predictionSettings']['threshold'])
            if idx != -1:
               return float(100 - idx)
            else:
               self.logger.warning('Binary search found no solution, try linear search')
               qrf1_percentiles = [qrf.predict(self.input_df, quantile=q) for q in np.linspace(1.0, 99.0, 99)]
               for i in range(0, len(qrf1_percentiles)):
                   if qrf1_percentiles[i] > self.cfg['predictionSettings']['threshold']:
                       return float(100-(i+1))


    def predict(self, predictor_file):
        if self.do_prediction is True:
            # ngb; NGBoost
            # qrf_nw; QRF without weights
            # qrf_ww; QRF wit weights
            ngb, qrf_nw, qrf_ww = pickle.load(open(predictor_file, 'rb'))

            res_ngb = ngb.pred_dist(self.input_df)
            self.prob_over_limit = self.prob_overlimit(qrf_nw)
            self.logger.info('Performed prediction: model=%s ' % predictor_file)

            dps = []

            # Saving predicted value
            self.predicted_value = float(res_ngb.loc[0])
            self.perc_available_features = round(self.available_features*100/len(self.input_df.columns), 0)
            point = {
                'time': self.day_to_predict,
                'measurement': self.cfg['influxDB']['measurementForecasts'],
                'fields': dict(PredictedValue=float(self.predicted_value),
                               AvailableFeatures=float(self.perc_available_features),
                               ProbOverLimit=float(self.prob_over_limit)),
                'tags': dict(location=self.location['code'], case=self.forecast_type,
                             predictor=predictor_file.split(os.sep)[-1].split('.')[0].split('_')[-1])
            }
            dps.append(point)

            # Write results on InfluxDB
            self.influxdb_client.write_points(dps, time_precision='s')
        else:
            self.logger.error('Model %s can not perform prediction, some features cannot be surrogated' % predictor_file)
            self.predicted_value = None
            self.perc_available_features = None