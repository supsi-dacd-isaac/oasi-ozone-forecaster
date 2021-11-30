import json
import logging
import sys
import argparse


import pandas as pd
import pytz
import urllib3
import os
from influxdb import InfluxDBClient

dir_path = os.path.dirname(os.path.realpath(__file__))
path_parent = os.path.dirname(dir_path)
sys.path.insert(0, path_parent)

from classes.artificial_features import ArtificialFeatures

urllib3.disable_warnings()

if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # Configuration file
    # --------------------------------------------------------------------------- #
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", help="configuration file")
    arg_parser.add_argument("-t", help="type (MOR | EVE)")
    arg_parser.add_argument("-l", help="log file (optional, if empty log redirected on stdout)")
    args = arg_parser.parse_args()

    # Load the main parameters
    config_file = args.c
    if os.path.isfile(config_file) is False:
        print('\nATTENTION! Unable to open configuration file %s\n' % config_file)
        sys.exit(1)

    cfg = json.loads(open(args.c).read())

    # Load the connections parameters and update the config dict with the related values
    cfg_conns = json.loads(open(cfg['connectionsFile']).read())
    cfg.update(cfg_conns)

    # Define the forecast type
    forecast_type = args.t

    # --------------------------------------------------------------------------- #
    # Set logging object
    # --------------------------------------------------------------------------- #
    if not args.l:
        log_file = None
    else:
        log_file = args.l

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=log_file)

    logger.info('Starting program')

    logger.info('Connection to InfluxDb server on socket [%s:%s]' % (cfg['influxDB']['host'], cfg['influxDB']['port']))
    try:
        influx_client = InfluxDBClient(host=cfg['influxDB']['host'], port=cfg['influxDB']['port'],
                                       password=cfg['influxDB']['password'], username=cfg['influxDB']['user'],
                                       database=cfg['influxDB']['database'], ssl=cfg['influxDB']['ssl'])
    except Exception as e:
        logger.error('EXCEPTION: %s' % str(e))
        sys.exit(3)
    logger.info('Connection successful')

    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)


    # --------------------------------------------------------------------------- #
    # Functions
    # --------------------------------------------------------------------------- #

    def get_results(AF, signals):
        results = []
        for sig in signals:
            results.append(AF.analyze_signal(sig))
        return results


    # --------------------------------------------------------------------------- #
    # Start testing
    # --------------------------------------------------------------------------- #

    # Test nr. 1: VOC_Totale

    # MOR, Measured VOC, 2021
    cfg['dayToForecast'] = '2021-06-20'
    cfg['VOC']['useCorrection'] = True
    cfg['VOC']['emissionType'] = 'measured'
    signals = ['VOC_Totale']
    assert get_results(AF, signals) == [39708.13241934641]

    # MOR, Forecasted VOC, 2021, use correction
    cfg['dayToForecast'] = '2021-06-20'
    cfg['VOC']['useCorrection'] = True
    cfg['VOC']['emissionType'] = 'forecasted'
    signals = ['VOC_Totale']
    assert get_results(AF, signals) == [41613.07999960428]

    # MOR, Forecasted VOC, 2021, don't use correction
    cfg['dayToForecast'] = '2021-06-20'
    cfg['VOC']['useCorrection'] = False
    cfg['VOC']['emissionType'] = 'forecasted'
    signals = ['VOC_Totale']
    assert get_results(AF, signals) == [50360.66693265781]

    # MOR, Measured VOC, 2019
    cfg['dayToForecast'] = '2019-06-20'
    cfg['VOC']['useCorrection'] = True
    cfg['VOC']['emissionType'] = 'measured'
    signals = ['VOC_Totale']
    assert get_results(AF, signals) == [39820.213070537204]

    # MOR, Forecasted VOC, 2019, use correction
    cfg['dayToForecast'] = '2019-06-20'
    cfg['VOC']['useCorrection'] = True
    cfg['VOC']['emissionType'] = 'forecasted'
    signals = ['VOC_Totale']
    assert get_results(AF, signals) == [41766.71981682944]

    # MOR, Forecasted VOC, 2019, don't use correction
    cfg['dayToForecast'] = '2019-06-20'
    cfg['VOC']['useCorrection'] = False
    cfg['VOC']['emissionType'] = 'forecasted'
    signals = ['VOC_Totale']
    assert get_results(AF, signals) == [50577.91145583987]

    forecast_type = 'EVE'
    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)

    # EVE, Forecasted VOC, 2021, use correction
    cfg['dayToForecast'] = '2021-06-20'
    cfg['VOC']['useCorrection'] = True
    cfg['VOC']['emissionType'] = 'forecasted'
    signals = ['VOC_Totale']
    assert get_results(AF, signals) == [51081.18809719044]

    # MOR, Forecasted VOC, 2019, use correction
    cfg['dayToForecast'] = '2019-06-20'
    cfg['VOC']['useCorrection'] = True
    cfg['VOC']['emissionType'] = 'forecasted'
    signals = ['VOC_Totale']
    assert get_results(AF, signals) == [36253.35103891304]

    # Test nr. 2 KLO-LUG forecasted gradient in 2021

    forecast_type = 'MOR'
    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    cfg['dayToForecast'] = '2021-07-17'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [7.3180588235294275, 1.0]
    cfg['dayToForecast'] = '2021-07-16'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [5.80655882352934, 0.0]

    # KLO-LUG measured gradient before 2021

    cfg['dayToForecast'] = '2017-07-16'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [3.385517241379489, 0.0]
    cfg['dayToForecast'] = '2017-07-17'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [-1.142758620689733, 0.0]

    forecast_type = 'EVE'
    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    cfg['dayToForecast'] = '2021-07-16'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [6.626441176470398, 1.0]
    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    cfg['dayToForecast'] = '2021-07-17'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [6.850470588235476, 1.0]
    cfg['dayToForecast'] = '2021-07-18'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [5.77594117647066, 0.0]

    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    cfg['dayToForecast'] = '2017-07-16'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [3.385517241379489, 0.0]
    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    cfg['dayToForecast'] = '2017-07-17'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [-1.142758620689733, 0.0]
    cfg['dayToForecast'] = '2017-07-18'
    signals = ['KLO-LUG', 'KLO-LUG_favonio']
    assert get_results(AF, signals) == [-2.998620689655013, 0.0]

    # Test nr. 3 Other signals

    signals = ['P_BIO__T_2M__MAX', 'TICIA__T_2M__12h_mean', 'TICIA__T_2M__12h_mean_squared', 'P_BIO__TD_2M__transf',
               'P_BIO__TOT_PREC__sum', 'OTL__GLOB__mean_mor', 'OTL__GLOB__mean_eve', 'BIO__CN__48h__mean',
               'LUG__O3__24h__mean', 'MS-LUG__T__24h__mean', 'NOx_Totale']

    forecast_type = 'MOR'
    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    cfg['dayToForecast'] = '2019-08-01'
    assert get_results(AF, signals) == [31.599999999999966, 29.39999999999992, 864.3599999999954, 52734.375, 37.6,
                                        265.1142857142857, 360.2909090909091, 2.3157894736842106, 82.85341974468085,
                                        24.079856115107912, 8196.797378]

    forecast_type = 'EVE'
    AF = ArtificialFeatures(influx_client, forecast_type, cfg, logger)
    cfg['dayToForecast'] = '2019-07-10'
    assert get_results(AF, signals) == [28.899999999999977, 26.749999999999943, 715.5624999999969, 49027.89599999986, 0.0,
                                        241.075, 313.94545454545454, 2.8421052631578947, 94.70607321385475,
                                        23.371942446043168, 8395.299337]

    logger.info('Ending program')
