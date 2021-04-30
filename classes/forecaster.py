# import section
import json
import pandas as pd
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

    def build_model_input_dataset(self, input_signals, day_to_predict, input_cfg_file):
        """
        Build the dataset
        """
        self.day_to_predict = day_to_predict
        self.cfg_signals = json.loads(open(input_cfg_file).read())

        input_data_values = []
        for signal in self.cfg_signals['signals']:
            # Take into account only the inputs needed by the model
            if signal in self.cfg_signals['signals']:
                input_data_values.append(input_signals[signal])

        self.logger.info('Create the input dataframe')
        self.input_df = pd.DataFrame([input_data_values], columns=self.cfg_signals['signals'],
                                     index=[pd.DatetimeIndex([self.day_to_predict*1e9])])


    def predict(self, predictor_file):
        model = pickle.load(open(predictor_file, 'rb'))
        res = model.pred_dist(self.input_df)
        self.logger.info('Predictor stored in %s -> Forecast: %.1f' % (predictor_file, res.loc[0]))
