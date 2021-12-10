Configuration files
===================

In the following we describe the main sections of the configuration file and the configurable parameters therein

Dataset creator settings
************************

*  ``datasetCreator``: There are three possible ways to create a dataset, hence three possible parameters:

  *  ``regions``: use the parameters configured in ``regions`` and ``measuredSignalsStations`` below to create a list of signals and download them
  *  ``customJSON``: use a list of signals configured in ``customJSONSignals`` below and download them
  *  ``CSVreader``: load a list of CSV files defined in ``loadCsvFolder`` below
  
*  ``saveDataset``: if ``true``, save the downloaded or loaded dataset in the ``outputCsvFolder``
*  ``loadSignalsFolder``: path to the folder where the features JSON files are stored
*  ``customJSONSignals``: list of dictionaries containing:

   *  a JSON file containing the features
   *  a list of one or more columns of the dataset, containing the O3 values of the previous day. If more than one coulmn is provided, the maximum of the daily values will constitute the response vector Y

   .. code-block:: python
   
      {
        "filename": "CHI_MOR.json"
        "targetColumn": ["CHI__YO3__d1", "BIO__YO3__d1"]
      } 
  
*  ``loadCsvFolder``: path to the folder containind the CSV dataset files
*  ``csvFiles``: a list of dictionaries containing:

   *  a CSV file containing the whole dataset
   *  a list of one or more columns of the dataset, containing the O3 values of the previous day. If more than one coulmn is provided, the maximum of the daily values will constitute the response vector Y

   .. code-block:: python
   
      {
        "filename": "BIO_MOR.csv"
        "targetColumn": ["CHI__YO3__d1", "BIO__YO3__d1"]
      }
 
*  ``outputCsvFolder``: path where the downloaded datasets and selected features are saved
*  ``outputSignalFolder``: path where the JSON file of regional signals are saved
*  ``startDay``: "mm-dd" format date, the start of the period to download each year. Ex: "05-15"
*  ``endDay``: "mm-dd" format date, the end of the period to download each year. Ex: "09-30"
*  ``years``: list of years for which the data are downloaded, between ``startDay`` and ``endDay``. Ex: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
*  ``sleepTimeBetweenQueries``: Sleeping time in seconds between each query to InfluxDB

Regions settings
****************

*  ``regions``: a dictionary of regions. A region is a dictionary itself, composed by measure stations, forecast stations and a list of one or more target columns. For instance:

   .. code-block:: python

      "Bioggio":
      {
        "MeasureStations": ["BIO", "LUG", "MS-LUG"],
        "targetColumn": ["BIO__YO3__d1"],
        "ForecastStations": ["P_BIO"]
      }
      
*  ``measuredSignalsStations``: list of signals that are measured at each measurement station. For instance:

   .. code-block:: python

      "measuredSignalsStations":
      {
        "CHI": ["CN", "Gl", "NO", "NO2", "NOx", "O3", "P", "Prec", "RH", "T", "WD", "WS"],
        "MEN": ["CN", "Gl", "NO", "NO2", "NOx", "O3", "P", "Prec", "RH", "T", "WDvect", "WSvect"],
        "LUG": ["NO", "NO2", "NOx", "O3"],
      }

*  ``forecastedSignalsStations``: list of signals that are forecasted at each forecast station. For instance:

   .. code-block:: python

      "forecastedSignalsStations":
      {
        "TICIA": ["GLOB", "PS", "TOT_PREC", "RELHUM_2M", "T_2M", "TD_2M", "DD_10M", "FF_10M", "CLCT"],
        "P_BIO": ["GLOB", "PS", "TOT_PREC", "RELHUM_2M", "T_2M", "TD_2M", "DD_10M", "FF_10M", "CLCT"],
      }

*  ``allMeasuredSignals``: list of all measured signals. Do not modify
*  ``allForecastedSignals``: list of all forecasted signals. Do not modify

VOC settings
************

Configuration parameters for the calculation of wood VOC signal 

*  ``useCorrection``: if ``true``, apply a linear correction to the values calculated with the forecasted data with the following parameters:

   *  ``correction``: use the following two parameters to apply a correction through a regression line fit 
  
	  *  ``slope``: slope of the linear regression fitting the forecasted data
	  *  ``intercept``: intercept of the linear regression fitting the forecasted data
	
  *  ``emissionType``: if ``"forecasted"``, try to use forecasted data wherever possible, otherwise use measured data. If ``"measured"``, use measured data everywhere to calculate woods VOC
  
Feature analyzer settings
*************************

Configuration parameters to perform the Feature Selection using SHAP and NGBoost
    
*  ``numberEstimatorsNGB``: number of boosting iterations in NGB (see the `online documentation <https://github.com/stanfordmlgroup/ngboost/blob/master/ngboost/api.py/>`_)
*  ``learningRate``: the learning rate eta
*  ``numberSelectedFeatures``: number of most important features to be saved 
*  ``w1``: weight of O3 observations above ``threshold1`` :math:`\mu g/m^3`
*  ``w2``: weight of O3 observations between ``threshold2`` and ``threshold1`` :math:`\mu g/m^3`
*  ``w3``: weight of O3 observations between ``threshold3`` and ``threshold2`` :math:`\mu g/m^3`
*  ``threshold1``: highest threshold limit, currently 240 :math:`\mu g/m^3`
*  ``threshold2``: intermediate threshold limit, currently 180 :math:`\mu g/m^3`
*  ``threshold3``: lowest threshold limit, currently 135 :math:`\mu g/m^3`
  
Grid search settings
********************

*  ``w1_start``: starting value for the weight :math:`w_1` in the grid search
*  ``w1_end``: last value for the weight :math:`w_1` in the grid search
*  ``w1_step``: step value for the weight :math:`w_1` in the grid search
*  ``w2_start``: starting value for the weight :math:`w_2` in the grid search
*  ``w2_end``: last value for the weight :math:`w_2` in the grid search
*  ``w2_step``: step value for the weight :math:`w_2` in the grid search
*  ``w3_start``: starting value for the weight :math:`w_3` in the grid search
*  ``w3_end``: last value for the weight :math:`w_3` in the grid search
*  ``w3_step``: step value for the weight :math:`w_3` in the grid search
*  ``typeGridSearch``: there are three possible ways to perform a grid search, as explained in :ref:`Grid search`:

   *  ``multiple``: perform a feature selection inside each fold
   *  ``single``: perform one feature selection on the whole dataset and use the selecetd feazures on each fold
   *  ``test``: mostly used only for testing, perform a feature selection on the whole dataset and then divide the dataset in 80% training and 20% testing
