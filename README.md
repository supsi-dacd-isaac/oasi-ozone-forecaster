# oasi-ozone-forecaster


## Main scripts:

### `features_selector.py`: Create dataset and (optionally) perform feature selection:
```
venv/bin/python3 features_selector.py -c conf/oasi_datasets.json -t MOR -l logs/datasets.log
venv/bin/python3 features_selector.py -c conf/oasi_datasets.json -t EVE -l logs/datasets.log
```

For the enabling/disabling of the feature selection the flag `performFeatureSelection` has to be set (see below).


## Main sections and parameters of JSON configuration file:

- `connectionsFile`: JSON file with InfluxDB connection coordinates
- `forecastPeriod`
  - `case`: if current, use today's date, else use `startDate` and `endDate`
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
	"filename": JSON file containing the features. Ex: "CHI_MOR.json".
	"targetColumn": list of one or more columns of the dataset, containing the O3 values of the previous day. If more than one coulmn is provided, the maximum of the daily values will constitute the response vector Y. Ex: ["CHI__YO3__d1", "BIO__YO3__d1"].
	}
	```
  - `loadCsvFolder`: path to the folder containind the CSV dataset files
  - `csvFiles`: list of dictionaries of the following shape
    ```yaml
	{
	"filename": CSV file containing the dataset. Ex: "CHI_MOR.csv".
	"targetColumn": list of one or more columns of the dataset, containing the O3 values of the previous day. If more than one coulmn is provided, the maximum of the daily values will constitute the response vector Y. Ex: ["CHI__YO3__d1", "BIO__YO3__d1"].
	}
	```  
  - `outputCsvFolder`: path where the downloaded datasets and selected features are saved
  - `outputSignalFolder`: path where the JSON file of regional signals are saved
  - `startDay`: "mm-dd" format date, the start of the period to download each year. Ex: "05-15"
  - `endDay`: "mm-dd" format date, the end of the period to download each year. Ex: "09-30"
  - `years`: list of years for which the data are downloaded, between `startDay` and `endDay`. Ex: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
  - `sleepTimeBetweenQueries`: Sleeping time in seconds between each query to InfluxDB
  - `regions`: a dictionary of regions. A region is a dictionary itself, composed by measure stations, forecast stations and a list of one or more target columns. Ex:
    ```yaml
	"Bioggio": 
	{
      "MeasureStations": ["BIO", "LUG", "MS-LUG"],
      "targetColumn": ["BIO__YO3__d1"],
      "ForecastStations": ["P_BIO"]
    }
	```  
  - `measuredSignalsStations`: list of signals that are measured at each measurement station
  - `forecastedSignalsStations`: list of signals that are forecasted at each forecast station
  - `allMeasuredSignals`: list of all measured signals
  - `allForecastedSignals`: list of all forecasted signals
  - `VOC`: configuration parameters for the calculation of wood VOC signal 
    - `useCorrection`: if `true`, apply a linear correction to the values calculated with the forecasted data with the following parameters
    - `correction`
	  - `slope`: slope of the linear regression fitting the forecasted data
	  - `intercept`: intercept of the linear regression fitting the forecasted data
    - `emissionType`: if `"forecasted"`, try to use forecasted data wherever possible, otherwise use measured data. If `"measured"`, use measured data everywhere to calculate woods VOC
  - `featuresAnalyzer`: configuration parameters to oerform the Feature Selection using SHAP and NGBoost
    - `performFeatureSelection`: if `True`, perform feature selection after loading/downloading the data 
    - `datasetCreator`: 3 possible values
      - `regions`: use the values configured in `regions` and `measuredSignalsStations` above to create a list of signals and download them
      - `customJSON`: use a list of signals configured in `customJSONSignals` and download them
      - `CSVreader`: load a list of CSV files defined in `loadCsvFolder` above
    - `numberEstimatorsNGB`: number of boosting iterations in NGB (see: [documentation](https://github.com/stanfordmlgroup/ngboost/blob/master/ngboost/api.py))
    - `learningRate`: the learning rate eta
    - `numberSelectedFeatures`: number of most important features to be saved 
    - `w1`: weight of O3 observations above 240 ug/m^3
    - `w2`: weight of O3 observations between 180 and 240 ug/m^3
    - `w3`: weight of O3 observations between 135 and 180 ug/m^3


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

## Internal instructions:

1. Beware that the `skgarden` package is severely outdated, thus some manual modifications are necessary in folder `/venv/lib64/python3.8/site-packages/skgarden/`:
   - file `quantile/ensemble.py`, line 40: add `sample_weight` to method arguments
   - file `quantile/ensemble.py`, line 79: add `sample_weight` to method call
   - file `quantile/tree.py`, lines 221 and 230: remove `presort=False` option for clean output (optional)
   - file `mondrian/ensemble/forest.py`, line 96, 229: add `sample_weight` to method arguments
   - file `mondrian/ensemble/forest.py`, line 121, 254: add `sample_weight` to method call
2. Create the dataset you're interested in using the script `dataset_creator` on a Ununpentio connected machine, such as `dev03` for instance. Be sure to correctly set the parameters in the `datasetSettings` section of config file.
3. Move the thusly created dataset to some powerful machine (e.g. Shiva or Bombolo) and perform a computationally intensive grid search with the script `grid_search` over a configured set of weights.
   Be sure to have set the parameteres in the `gridSearcher` section of the config file, in particular `cfg["featuresAnalyzer"]["datasetCreator"]` should be set to `CSVreader`, the previously created dataset should be placed in the right `conf/csv/` folder and the file names should appear in the `cfg['datasetSettings']['csvFiles']` section.
   It is possible the machine will be angered by the presence of the package `skgarden`. In this case you should comment the offending lines of code, which are not needed anyway for the grid search.
4. Observe the output of the grid search and manually select the best combination of weights. Do update the config file accordingly.
5. On a machine able to run the whole code, use the script `final_model_creator` with the previously created dataset and the previously selected weights to create the final models, which can be used to perform the daily forecast.
