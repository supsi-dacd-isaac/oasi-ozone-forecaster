Main scripts
============

In the following we describe the main scripts of and how they work

Dataset creator
***************

The script ``dataset_creator.py``, when executed, reads the information contained in the configuration files and creates a dataset witht he specified features and time range in a folder in the directory ``output/``.

Relevant parameters are listed in the :ref:`Dataset creator settings` section

To execute, use the commands

.. code-block:: python

   venv/bin/python3 dataset_creator.py -c conf/oasi_datasets.json -t MOR -l logs/datasets.log
   venv/bin/python3 dataset_creator.py -c conf/oasi_datasets.json -t EVE -l logs/datasets.log

Features selector
*****************

The script ``features_selector.py`` creates the dataset as before, and in addition it performs a feature selection according to the parameters in the configuration file

Relevant parameters are listed in the :ref:`Feature analyzer settings` section

To execute, use the commands

.. code-block:: python

   venv/bin/python3 features_selector.py -c conf/oasi_datasets.json -t MOR -l logs/datasets.log
   venv/bin/python3 features_selector.py -c conf/oasi_datasets.json -t EVE -l logs/datasets.log

Grid search
***********

The script ``grid_search.py`` creates the dataset and then performs a grid search on the set of weights :math:`\{w_1, w_2, w_3 \}` over the intervals specified in the configuration files. 
The cross validation divides the dataset in years, that is, in the first fold the data will be trained on years 2016-2021 to predict 2015, then in the second fold data will be trained on the years 2015, 2017-2021 to predict 2016, and so on.

For every combination of weights analyzed, it is either possible to:

* Perform one general feature selection on all the dataset which will be applied to all the folds in the cross validation.
* Perform a specific feature selection inside each fold of the cross validation.

Relevant parameters are listed in the :ref:`Grid search settings` section

To execute, use the commands

.. code-block:: python

   venv/bin/python3 grid_search.py -c conf/oasi_datasets.json -t MOR -l logs/datasets.log
   venv/bin/python3 grid_search.py -c conf/oasi_datasets.json -t EVE -l logs/datasets.log

Final model creator
*******************

The script ``final_model_creator.py``, given a set of weights :math:`\{w_1, w_2, w_3 \}` in the configuration files, will create the final NGBoost and QRF models. The first is used for the daily single value prediction, while the second for the probabilistic prediction through the quantiles.

To execute, use the commands

.. code-block:: python

   venv/bin/python3 final_model_creator.py -c conf/oasi_datasets.json -t MOR -l logs/datasets.log
   venv/bin/python3 final_model_creator.py -c conf/oasi_datasets.json -t EVE -l logs/datasets.log
