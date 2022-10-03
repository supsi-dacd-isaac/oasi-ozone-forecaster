Unit testing
============

A few tests are provided to ensure everything was installed correctly. These are already preset and the configuration file should not be modified unless you know exactly what you're doing

To perform these tests singularly you can use the commands

.. code-block:: python

   venv/bin/python3 tests/test_artificial_features.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
   venv/bin/python3 tests/test_features_analyzer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
   venv/bin/python3 tests/test_input_gatherer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
   venv/bin/python3 tests/test_grid_search.py -c conf/oasi_tests.json -t MOR -l logs/tests.log
   venv/bin/python3 tests/test_model_trainer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log

Or alternatively you can run them all in sequence with the command 

.. code-block:: python

   venv/bin/python3 tests/perform_all_tests.py -l logs/tests.log
