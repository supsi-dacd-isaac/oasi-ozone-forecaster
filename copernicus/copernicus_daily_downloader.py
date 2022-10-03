import argparse
import cdsapi
import json
import logging
import os
import sys
import time

# from copernicus_data_fetcher import SingleMetavarHelpFormatter
from datetime import date, timedelta


def main(conf, logger):
    params = conf['params']
    starting_date = date.today() - timedelta(hours=conf['hoursBack'])
    params['date'] = '{}/{}'.format(starting_date, date.today())
    c = cdsapi.Client()
    areas_to_vars = conf['areas_to_vars']
    for area in areas_to_vars:
        logger.info('Requesting data for region \'%s\', interval: %s' % (area, params['date']))
        params['area'] = areas_to_vars[area]['coord']
        params['variable'] = areas_to_vars[area]['variables']
        retries = 0
        while retries < conf['max_retries']:
            try:
                c.retrieve(conf['dataset'], params, '{}-{}-{}.zip'.format(os.path.join(conf['output_dir'], conf['output_file']), area, starting_date))
                logger.info('Region \'%s\' completed' % area)
                break
            except Exception as e:
                logger.error('failed to retrieve data for area {}: {}'.format(area, str(e)))
                retries += 1


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--conf", help="use the specified config file")
    arg_parser.add_argument("-l", "--log", help="use the specified log file")
    args = vars(arg_parser.parse_args())

    conf_arg = args['conf']
    try:
        if os.path.isfile(conf_arg) is False:
            print('[ERROR] failed to open provided configuration file', conf_arg)
            sys.exit(1)
    except Exception as e:
        print('[ERROR] failed to check whether provided config file exists:', str(e))
        sys.exit(1)

    try:
        config = json.loads(open(args['conf']).read())
    except Exception as e:
        print('failed to deserialize json config: ', str(e))
        sys.exit(1)

    log_file = args['log']
    file_name = None if not log_file else log_file
    log = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=log_file)

    sys.exit(main(config, log))
