.. _ref_main_scripts:

Main scripts
============

In the following we describe the main scripts of and how they work

Dataset creator
***************

``dataset_creator.py`` when executed reads the information contained in the configuration files and creates a dataset witht he specified features and time range in a folder in the directory ``output/``.
To execute, launch the commands

.. code-block:: python

   venv/bin/python3 dataset_creator.py.py -c conf/oasi_datasets.json -t MOR -l logs/datasets.log
   venv/bin/python3 dataset_creator.py.py -c conf/oasi_datasets.json -t EVE -l logs/datasets.log

Features selector
*****************
