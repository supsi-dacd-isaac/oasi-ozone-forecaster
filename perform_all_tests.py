import logging
import os
import argparse
import time

if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # Configuration file
    # --------------------------------------------------------------------------- #
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", help="log file (optional, if empty log redirected on stdout)")
    args = arg_parser.parse_args()

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

    # --------------------------------------------------------------------------- #
    # Functions
    # --------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------- #
    # Start calculations
    # --------------------------------------------------------------------------- #

    start_time = time.time()

    os.system('venv/bin/python3 tests/test_artificial_features.py -c conf/oasi_tests.json -t MOR -l logs/tests.log')
    os.system('venv/bin/python3 tests/test_features_analyzer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log')
    os.system('venv/bin/python3 tests/test_input_gatherer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log')
    os.system('venv/bin/python3 tests/test_grid_search.py -c conf/oasi_tests.json -t MOR -l logs/tests.log')
    os.system('venv/bin/python3 tests/test_model_trainer.py -c conf/oasi_tests.json -t MOR -l logs/tests.log')

    logger.info("--- %s seconds elapsed to execute all tests ---" % (time.time() - start_time))

    logger.info('Ending program')
