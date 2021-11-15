# oasi-ozone-forecaster

## CURL examples to download forecasts results from ISAAC database:
```
curl -G 'http://$HOST:$PORT/query?u=$USER&p=$PWD&pretty=true' --data-urlencode "db=$DB" --data-urlencode "q=SELECT * FROM o3_forecasts_results WHERE flag_best='true' and time>='2021-05-13' and time<='2021-05-14' GROUP BY location, case"
curl -G 'http://$HOST:$PORT/query?u=$USER&p=$PWD&pretty=true' --data-urlencode "db=$DB" --data-urlencode "q=SELECT * FROM o3_forecasts_probabilities WHERE flag_best='true' and time>='2021-05-13' and time<='2021-05-14' GROUP BY location, case, interval"
```

## Execute unit tests:

```
export PYTHONPATH=/home/isaac-user02/run/python/oasi-ozone-forecaster
venv/bin/python3 tests/test_input_gatherer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
venv/bin/python3 tests/test_features_analyzer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
venv/bin/python3 tests/test_artificial_features.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
```

## Create dataset and optionally perform feature selection:
```
venv/bin/python3 features_selector.py -c conf/oasi_datasets.json -t MOR -l logs/datasets.log
venv/bin/python3 features_selector.py -c conf/oasi_datasets.json -t EVE -l logs/datasets.log
```


## JSON sections and parameters:

- `connectionsFile`: JSON file with InfluxDB connection coordinates
- `forecastPeriod`
  - `case`: if current, use today's date, else use startDate and endDate
  - `startDate`: Start of forecasting period
  - `endDate`: End of forecasting period
- `dayToForecast`: 
- `predictionSettings`
  - `operationMode`
  - `distributionSamples`
  - `thresholds`
  - `startDateForMeanImputation`
- `datasetSettings`
  - `saveDataset`: if `true` save the downloaded dataset in the `outputCsvFolder`
  - `loadSignalsFolder`: path to the folder where the features JSON files are stores
  - `customJSONSignals`: list of dictionaries of the following shape
    ```yaml
	{
		"filename": "best_features_CHI_MOR.json", 
		"targetColumn": ["CHI__YO3__d1"]}
	} 
  - `loadCsvFolder`
  - `csvFiles`
  - `outputCsvFolder`
  - `outputSignalFolder`
  - `startDay`
  - `endDay`
  - `years`
  - `sleepTimeBetweenQueries`
  - ``
  - ``
  - ``
