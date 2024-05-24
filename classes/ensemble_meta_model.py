# import section
import numpy as np


class EnsembleMetaModel:
    """
    Generic interface for EnsembleMetaModel class
    """

    def __init__(self, inputs_gatherer, influxdb_client, cfg, logger):
        """
        Constructor
        """
        # Set the variables
        self.inputs_gatherer = inputs_gatherer
        self.influxdb_client = influxdb_client
        self.cfg = cfg
        self.logger = logger

    def rule_based_ensemble(self):
        # Set the day to forecast
        self.inputs_gatherer.set_forecast_day()
        day_to_predict_ns = int(self.inputs_gatherer.day_to_predict * 1e9)

        prediction_results = []

        # Cycle over the regions
        for region in self.cfg['regions'].keys():

            # Cycle over the derived forecaster that must be taken into account
            for dp_cfg in self.cfg['regions'][region]['forecaster']['derivedForecasters']:
                predicted_data = {}

                # Cycle over the output signals
                for output_signal in dp_cfg['cases'][self.inputs_gatherer.forecast_type]:

                    # Cycle over the families configured for the derived family
                    for family in dp_cfg['predictorFamilies']:
                        predicted_data[family] = {}

                        # Get data for all the configured models
                        for md_code in dp_cfg['modelsMeasurements'].keys():
                            md_measurement = dp_cfg['modelsMeasurements'][md_code]
                            if md_code == 'qrf':
                                predicted_data = self.get_qrf_data_prediction(predicted_data, md_measurement, family,
                                                                              day_to_predict_ns, region, output_signal)

                            else:
                                predicted_data = self.get_data_prediction(predicted_data, md_measurement, family,
                                                                          day_to_predict_ns, region, output_signal,
                                                                          md_code)

                    # Get values configured to be aggregated
                    predicted_vals = self.get_data_to_aggregate_for_rule_base_ensemble(predicted_data, dp_cfg)

                    # Conservative approach: we have to be safe that every family-model couple has given its forecast
                    if len(predicted_vals) == len(dp_cfg['predictorFamilies']) * len(dp_cfg['predictorModels']):
                        self.logger.info('Calculate derived prediction, '
                                         'family %s, case [%s-%s-%s]' % (dp_cfg['name'], region,
                                                                         self.inputs_gatherer.forecast_type,
                                                                         output_signal))

                        # Aggregate values
                        agg_value = self.aggregate_values(predicted_vals, dp_cfg['aggregation'])

                        # Rule-based approach: it associates a prediction to a derived family
                        # depending on configured intervals

                        # Cycle over the intervals
                        for interval in dp_cfg['intervals']:
                            if interval['min'] <= agg_value < interval['max']:

                                # Cycle over the predictions to find the configured result
                                for kf in predicted_data:
                                    if kf == interval['chosenFamily']:
                                        # Set result dataset
                                        tmp_result = self.set_result_dataset(region, output_signal, dp_cfg['name'],
                                                                             predicted_data[kf])

                                        # Save prediction data in DB
                                        self.save_derived_forecast_in_db(predictions=tmp_result)

                                        # Append to results list
                                        prediction_results.append(tmp_result)
                    else:
                        self.logger.error('Unable to calculate derived prediction, family %s, '
                                          'case [%s-%s-%s]' % (dp_cfg['name'], region,
                                                               self.inputs_gatherer.forecast_type, output_signal))

        return prediction_results

    def save_derived_forecast_in_db(self, predictions):
        # Save data in the database
        dps = []
        # NGBoost section
        point = {
            'time': predictions['day_to_predict'],
            'measurement': self.cfg['influxDB']['measurementOutputSingleForecast'],
            'fields': dict(PredictedValue=float(predictions['ngb_prediction']),
                           AvailableFeatures=float(predictions['perc_available_features'])),
            'tags': dict(location=predictions['region'], case=predictions['forecast_type'],
                         flag_best=predictions['flag_best'], predictor=predictions['predictor'],
                         signal=predictions['output_signal'])
        }
        dps.append(point)

        # QRF section
        for interval in predictions['qrf_prediction']['thresholds'].keys():
            point = {
                'time': predictions['day_to_predict'],
                'measurement': self.cfg['influxDB']['measurementOutputThresholdsForecast'],
                'fields': dict(Probability=float(
                    predictions['qrf_prediction']['thresholds'][interval])),
                'tags': dict(location=predictions['region'], case=predictions['forecast_type'],
                             flag_best=predictions['flag_best'], predictor=predictions['predictor'],
                             signal=predictions['output_signal'], interval=interval)
            }
            dps.append(point)

        for quantile in predictions['qrf_prediction']['quantiles'].keys():
            point = {
                'time': predictions['day_to_predict'],
                'measurement': self.cfg['influxDB']['measurementOutputQuantilesForecast'],
                'fields': dict(PredictedValue=float(predictions['qrf_prediction']['quantiles'][quantile])),
                'tags': dict(location=predictions['region'], case=predictions['forecast_type'],
                             flag_best=predictions['flag_best'], predictor=predictions['predictor'],
                             signal=predictions['output_signal'], quantile=quantile)
            }
            dps.append(point)

        # XGBoost section
        point = {
            'time': predictions['day_to_predict'],
            'measurement': self.cfg['influxDB']['measurementOutputSingleForecastXGBoost'],
            'fields': dict(PredictedValue=float(predictions['xgb_prediction'])),
            'tags': dict(location=predictions['region'], case=predictions['forecast_type'],
                         flag_best=predictions['flag_best'], predictor=predictions['predictor'],
                         signal=predictions['output_signal'])
        }
        dps.append(point)

        # LightGBM section
        point = {
            'time': predictions['day_to_predict'],
            'measurement': self.cfg['influxDB']['measurementOutputSingleForecastLightGBM'],
            'fields': dict(PredictedValue=float(predictions['lgb_prediction'])),
            'tags': dict(location=predictions['region'], case=predictions['forecast_type'],
                         flag_best=predictions['flag_best'], predictor=predictions['predictor'],
                         signal=predictions['output_signal'])
        }
        dps.append(point)

        # Write results on InfluxDB
        try:
            self.influxdb_client.write_points(dps, time_precision='s')
            self.logger.info('Inserted %i points in InfluxDB' % len(dps))
        except Exception as e:
            self.logger.warning('EXCEPTION: %s' % str(e))
            self.logger.warning('Unable to insert results for regressors of derived predictor %s ' % predictions['predictor'])

    def get_data_to_aggregate_for_rule_base_ensemble(self, predicted_data, derived_predictor_cfg):
        predicted_vals = []
        for kf in predicted_data.keys():
            if kf in derived_predictor_cfg['predictorFamilies']:
                for pred_model in derived_predictor_cfg['predictorModels']:
                    try:
                        if (predicted_data[kf]['%s_prediction' % pred_model] is not None and
                                predicted_data[kf]['%s_prediction' % pred_model] != 0):
                            if 'qrf' in pred_model:
                                predicted_vals.append(
                                    predicted_data[kf]['%s_prediction' % pred_model]['quantiles']['perc50'])
                            else:
                                predicted_vals.append(predicted_data[kf]['%s_prediction' % pred_model])
                    except Exception as e:
                        self.logger.error('EXCEPTION: %s' % str(e))
        return predicted_vals

    def get_qrf_data_prediction(self, predicted_data, md_measurement, family, day_to_predict_ns, region, output_signal):
        predicted_data[family]['qrf_prediction'] = {'thresholds': {}, 'quantiles': {}}

        # Get quantiles data
        query = ("SELECT mean(PredictedValue) FROM %s_quantiles "
                 "WHERE time=%i AND location='%s' AND "
                 "signal='%s' AND case='%s' AND predictor=\'%s\' "
                 "GROUP BY quantile") % (md_measurement, day_to_predict_ns, region, output_signal,
                                         self.inputs_gatherer.forecast_type, family)
        res = self.influxdb_client.query(query, epoch='s')
        for q_data in res.raw['series']:
            predicted_data[family]['qrf_prediction']['quantiles'][q_data['tags']['quantile']] = q_data['values'][0][1]

        # Get intervals/thresholds data
        query = ("SELECT mean(Probability) FROM %s_thresholds "
                 "WHERE time=%i AND location='%s' AND "
                 "signal='%s' AND case='%s' AND predictor=\'%s\' "
                 "GROUP BY interval") % (md_measurement, day_to_predict_ns, region, output_signal,
                                         self.inputs_gatherer.forecast_type, family)
        self.logger.info("Query: %s" % query)
        res = self.influxdb_client.query(query, epoch='s')
        for p_data in res.raw['series']:
            predicted_data[family]['qrf_prediction']['thresholds'][p_data['tags']['interval']] = p_data['values'][0][1]

        return predicted_data

    def get_data_prediction(self, predicted_data, md_measurement, family, day_to_predict_ns, region, output_signal,
                            md_code):
        query = ("SELECT mean(PredictedValue) FROM %s "
                 "WHERE time=%i AND location='%s' AND "
                 "signal='%s' AND case='%s' AND predictor=\'%s\' ") % (md_measurement,
                                                                       day_to_predict_ns,
                                                                       region, output_signal,
                                                                       self.inputs_gatherer.forecast_type,
                                                                       family)
        self.logger.info("Query: %s" % query)
        try:
            pred_value = self.influxdb_client.query(query, epoch='s').raw['series'][0]['values'][0][1]
            predicted_data[family]['%s_prediction' % md_code] = pred_value
        except Exception as e:
            self.logger.error('Unable to get value from query: %s' % query)
            self.logger.error('Exception: %s' % str(e))
        return predicted_data

    def set_result_dataset(self, region, output_signal, family_name, models_predictions):
        res = {
            'day_to_predict': self.inputs_gatherer.day_to_predict,
            'output_signal': output_signal,
            'region': region,
            'forecast_type': self.inputs_gatherer.forecast_type,
            'predictor': family_name,
            'perc_available_features': 100.0,
            'ngb_prediction': models_predictions['ngb_prediction'],
            'lgb_prediction': models_predictions['lgb_prediction'],
            'xgb_prediction': models_predictions['xgb_prediction'],
            'qrf_prediction': models_predictions['qrf_prediction'],
            'flag_prediction': True,
            'flag_best': False,
            'best_predictor': None
        }

        # Check if the derived predictor is configured as "best"
        best_predictors_cfg = self.cfg['regions'][region]['forecaster']['bestLabels'][self.inputs_gatherer.forecast_type]
        best_family = best_predictors_cfg[output_signal]['family']
        if best_family is not None and best_family == res['predictor']:
            res['flag_best'] = True
            res['best_predictor'] = best_predictors_cfg[res['output_signal']]['predictor']

        return res

    def aggregate_values(self, vals, agg_func):
        # Currently only mean function is available
        if agg_func == 'mean':
            return np.mean(np.array(vals))
        else:
            return np.mean(np.array(vals))

