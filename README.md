# oasi-ozone-forecaster [![Documentation Status](https://readthedocs.org/projects/oasi-ozone-forecaster/badge/?version=latest)](https://oasi-ozone-forecaster.readthedocs.io/en/latest/?badge=latest)

Please refer to the [Read the Docs documentation](https://oasi-ozone-forecaster.readthedocs.io/en/latest/) for a description of the configurations and classes. 


## Utilities:

### CURL examples to download forecasts results from ISAAC database:
```
curl -G 'http://$HOST:$PORT/query?u=$USER&p=$PWD&pretty=true' --data-urlencode "db=$DB" --data-urlencode "q=SELECT * FROM o3_forecasts_results WHERE flag_best='true' and time>='2021-05-13' and time<='2021-05-14' GROUP BY location, case"
curl -G 'http://$HOST:$PORT/query?u=$USER&p=$PWD&pretty=true' --data-urlencode "db=$DB" --data-urlencode "q=SELECT * FROM o3_forecasts_probabilities WHERE flag_best='true' and time>='2021-05-13' and time<='2021-05-14' GROUP BY location, case, interval"
```

### Execute unit tests:

```
export PYTHONPATH=/home/isaac-user02/run/python/oasi-ozone-forecaster
venv/bin/python3 tests/test_input_gatherer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
venv/bin/python3 tests/test_features_analyzer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
venv/bin/python3 tests/test_artificial_features.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
venv/bin/python3 tests/test_grid_search.py -c conf/oasi_tests.json -t MOR -l logs/tests.log'
venv/bin/python3 tests/test_model_trainer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log'
```

or alternatively you can run 

```
venv/bin/python3 perform_all_tests.py -l logs/tests.log
```


## Particular notes:

- The base VOC and NOx signals should be calculated every year for the upcoming year. The baseline value is obtained from inferring a linear progression of the past years values, however, woods VOC needs to be calculated day by day. The code takes care of this last part, but the base value should be present on InfluxDB
