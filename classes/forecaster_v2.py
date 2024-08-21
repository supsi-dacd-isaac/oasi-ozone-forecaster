import json
import logging
import os
import glob
import pickle
import pandas as pd

from classes.features_analyzer_v2 import FeaturesAnalyzerV2
from classes.inputs_gatherer_v2 import InputsGathererV2
from classes.optimized_model_creator_v2 import OptimizedModelCreatorV2

class ForecasterV2:
    def __init__(self, ifc_df, ifc, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.ig = InputsGathererV2(ifc_df, cfg, logger, None)
        self.influx_client = ifc
        self.fa = FeaturesAnalyzerV2(self.ig, cfg, logger)
        self.predictors = {}
        self.input_predictor_structure = {}

    def retrieve_predictors(self):
        self.predictors = {}
        self.input_predictor_structure = {}

        for pred_metadata_file in glob.glob('%s/*___main_cfg.json' % self.cfg['folders']['models']):
            k_predictor = pred_metadata_file.split(os.sep)[-1].replace('__out___main_cfg.json', '')
            self.predictors[k_predictor] = {}
            self.predictors[k_predictor]['metadata'] = json.loads(open(pred_metadata_file).read())

            pred_model_file = pred_metadata_file.replace('main_cfg.json', 'predictor.pkl')
            lgb_model = pickle.load(open(pred_model_file, 'rb'))
            self.predictors[k_predictor]['lgbModel'] = lgb_model

            io_model_file = pred_metadata_file.replace('main_cfg.json', 'signals.json')
            self.predictors[k_predictor]['io'] = json.loads(open(io_model_file).read())

            self.input_predictor_structure[k_predictor] = self.build_cfg_structure(self.predictors[k_predictor]['io']['input'])


    @staticmethod
    def build_cfg_structure(input_list):
        input_cfg_structure = {
            'measures': { 'singletons': {}, "aggregations": {}},
            'forecast': { 'singletons': {}, "aggregations": {}}
        }
        for model_input in input_list:
            (location, signal_code, signal_type, case) = model_input.split('__')
            cfg_code = '%s__%s__%s' % (location, signal_code, signal_type)
            if signal_type == 'meas':
                if 'agg' in case:
                    (_, days_back_str, func) = case.split('_')
                    cfg_code_agg = '%s__func_%s' % (cfg_code, func)
                    days_back = int(days_back_str.replace('db', ''))
                    if cfg_code not in input_cfg_structure['measures']['aggregations'].keys():
                        input_cfg_structure['measures']['aggregations'][cfg_code_agg] = {'backDays': []}
                    input_cfg_structure['measures']['aggregations'][cfg_code_agg]['backDays'].append(days_back)
                else:
                    if cfg_code not in input_cfg_structure['measures']['singletons'].keys():
                        input_cfg_structure['measures']['singletons'][cfg_code] = {'backHours': []}
                    input_cfg_structure['measures']['singletons'][cfg_code]['backHours'].append(int(case.replace('sb', '')))
            else:
                if 'agg' in case:
                    (_, days_forward_str, func) = case.split('_')
                    cfg_code_agg = '%s__func_%s' % (cfg_code, func)
                    days_forward = int(days_forward_str.replace('df', ''))
                    if cfg_code not in input_cfg_structure['forecast']['aggregations'].keys():
                        input_cfg_structure['forecast']['aggregations'][cfg_code_agg] = {'forwardDays': []}
                    input_cfg_structure['forecast']['aggregations'][cfg_code_agg]['forwardDays'].append(days_forward)
                else:
                    if cfg_code not in input_cfg_structure['forecast']['singletons'].keys():
                        input_cfg_structure['forecast']['singletons'][cfg_code] = {'forwardHours': []}
                    input_cfg_structure['forecast']['singletons'][cfg_code]['forwardHours'].append(int(case.replace('sf', '')))
        return {'input': input_cfg_structure}



    def perform_forecast(self, curr_day):
        # Cycle over the hours to predict
        dps = []
        for k_predictor in self.predictors.keys():
            self.ig.retrieve_signals_from_files(k_predictor)
            self.fa = FeaturesAnalyzerV2(self.ig, self.cfg, self.logger)
            self.fa.dataset_creator(day_to_predict=curr_day)

            location = k_predictor.split('___')[1].split('__')[0]
            omc_v2 = OptimizedModelCreatorV2(location, self.input_predictor_structure[k_predictor], self.cfg, self.logger)
            omc_v2.io_dataset_creation(self.ig.predictor_input)
            omc_v2.order_dataset(self.ig.input_sequence[k_predictor])

            for hour_to_predict in self.cfg['forecastPeriod']['hours']:
                dt_to_predict_str = '%sT%02d:00:00Z' % (curr_day, hour_to_predict)
                dt_to_predict = pd.to_datetime(dt_to_predict_str, format='%Y-%m-%dT%H:%M:%SZ')
                dt_to_predict = dt_to_predict.tz_localize('UTC')
                row_to_predict = omc_v2.x_all.loc[[dt_to_predict]]
                pred_value = self.predictors[k_predictor]['lgbModel'].predict(row_to_predict)
                self.logger.info('PRED[(%s) - (%s)]: %.1f' % (k_predictor, dt_to_predict_str, pred_value[0]))

                (family, tmp) = k_predictor.split('___')
                (location, signal, tmp) = tmp.split('__')
                (_, sf) = tmp.split('_')
                point = {
                    'time': int(dt_to_predict.timestamp()),
                    'measurement': self.cfg['influxDB']['measurementHourlyForecast'],
                    'fields': dict(PredictedValue=float(pred_value)),
                    'tags': dict(predictor_family=family, location=location, signal=signal, step_forward=sf)
                }
                dps.append(point)
        if len(dps) > 0:
            self.logger.info('Sent %i points to InfluxDB server' % len(dps))
            self.influx_client.write_points(dps, time_precision=self.cfg['influxDB']['timePrecision'])
        else:
            self.logger.info('No data to send to InfluxDB server')