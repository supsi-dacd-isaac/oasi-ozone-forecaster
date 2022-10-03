import json
import logging
import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import urllib3

import warnings
warnings.filterwarnings("ignore")

def cfg_updating(main_cfg, cfg_case, label, region, output_signal):
    cfg_parent_file = '%s%s%s_%s_%s.json' % (main_cfg[cfg_case]['folder'], os.sep, label, region,
                                             main_cfg['predictorsFamily'])
    cfg_file = '%s%s%s_%s_%s_%s.json' % (main_cfg[cfg_case]['folder'], os.sep, label, region,
                                         main_cfg['predictorsFamily'], day_ahead)

    if os.path.exists(cfg_file):
        out_cfg = json.loads(open(cfg_file).read())
        tmp_cfg = out_cfg['regions'][region][main_cfg[cfg_case]['cfgSection']]['targetColumns'][target]
        tmp_cfg['weights'][pred_case] = {
            "w1": int(data['w1'][idx_min]),
            "w2": int(data['w2'][idx_min]),
            "w3": int(data['w3'][idx_min])
        }
        tmp_cfg['numberEstimatorsNGB'][pred_case] = int(data['ne'][idx_min])
        tmp_cfg['learningRateNGB'][pred_case] = float(data['lr'][idx_min])
    else:
        out_cfg = json.loads(open(cfg_parent_file).read())
        out_cfg['regions'][region][main_cfg[cfg_case]['cfgSection']]['targetColumns'] = dict()

        tmp_cfg = {
            "weights": {
                pred_case: {
                    "w1": int(data['w1'][idx_min]),
                    "w2": int(data['w2'][idx_min]),
                    "w3": int(data['w3'][idx_min])
                },
            },
            "numberEstimatorsNGB": {pred_case: int(data['ne'][idx_min])},
            "learningRateNGB": {pred_case: float(data['lr'][idx_min])}
        }
    out_cfg['regions'][region][main_cfg[cfg_case]['cfgSection']]['targetColumns'][target] = tmp_cfg
    if label == 'MT':
        out_cfg['regions'][region]['finalModelCreator']['targets'] = dict()
        out_cfg['regions'][region]['finalModelCreator']['targets'][target] = {'label': output_signal}
    json_obj = json.dumps(out_cfg, indent=2)
    with open(cfg_file, 'w') as outfile:
        outfile.write(json_obj)


if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # Configuration file
    # --------------------------------------------------------------------------- #
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", help="configuration file")
    args = arg_parser.parse_args()

    # Load the main parameters
    config_file = args.c
    if os.path.isfile(config_file) is False:
        print('\nATTENTION! Unable to open configuration file %s\n' % config_file)
        sys.exit(1)

    cfg = json.loads(open(args.c).read())

    print('Starting program')
    print('Weigths: %s' % cfg['kpiWeights'])
    results = []
    for target in cfg['targets']:
        print('Analisys of target %s' % target)
        data = None
        for data_file in sorted(glob.glob('%s%sGS_KPIs*%s*.csv' % (cfg['inputFolder'], os.sep, target))):
            print(data_file)
            if data is None:
                data = pd.read_csv(data_file)
            else:
                tmp_df = pd.read_csv(data_file)
                idx_df = pd.DataFrame(np.arange(len(data), len(data) + len(tmp_df)), columns=['idx'])
                tmp_df = pd.concat([tmp_df, idx_df], axis=1)
                tmp_df = tmp_df.set_index('idx')
                data = pd.concat([data, tmp_df])

        data = pd.concat([data, pd.DataFrame(np.arange(0,len(data)-1), columns=['idx'])], axis=1)

        # Get main metadata (region, case, target)
        tmp = data_file.split(os.sep)[-1].split('__')
        region = tmp[0].split('_')[-1]
        day_ahead = tmp[-1].split('_')[0]
        if 'MOR' in data_file.split(os.sep)[-2]:
            pred_case = 'MOR'
        else:
            pred_case = 'EVE'

        str_res = '%s,%s,%s,' % (region, pred_case, day_ahead)

        kpis = np.zeros(len(data))
        fw = open('%s%s%s.csv' % (cfg['outputFolder'], os.sep, target), 'w')
        fw.write('region,case,day_ahead,w1,w2,w3,ne,lr,Accuracy_1,Accuracy_2,Accuracy_3,Accuracy,RMSE1,RMSE2,RMSE3,RMSE,MAE1,MAE2,MAE3,MAE\n')
        for i in range(0, len(data)):
            kpi = 0
            for kpiw in cfg['kpiWeights']:
                kpi += cfg['kpiWeights'][kpiw] * data[kpiw][i]
            kpis[i] = kpi
            fw.write('%s,%s,%s,%i,%i,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (region, pred_case, day_ahead,
                                                                                        data['w1'][i],
                                                                                        data['w2'][i],
                                                                                        data['w3'][i],
                                                                                        data['ne'][i],
                                                                                        data['lr'][i],
                                                                                        data['Accuracy_1'][i],
                                                                                        data['Accuracy_2'][i],
                                                                                        data['Accuracy_3'][i],
                                                                                        data['Accuracy'][i],
                                                                                        data['RMSE1'][i],
                                                                                        data['RMSE2'][i],
                                                                                        data['RMSE3'][i],
                                                                                        data['RMSE'][i],
                                                                                        data['MAE1'][i],
                                                                                        data['MAE2'][i],
                                                                                        data['MAE3'][i],
                                                                                        data['MAE'][i]))

        idx_min = np.argmin(kpis)
        fw.write('BEST,,,,,,,,,,,,,,,,,,,\n')
        fw.write('%s,%s,%s,%i,%i,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (region, pred_case, day_ahead,
                                                                                    data['w1'][idx_min],
                                                                                    data['w2'][idx_min],
                                                                                    data['w3'][idx_min],
                                                                                    data['ne'][idx_min],
                                                                                    data['lr'][idx_min],
                                                                                    data['Accuracy_1'][idx_min],
                                                                                    data['Accuracy_2'][idx_min],
                                                                                    data['Accuracy_3'][idx_min],
                                                                                    data['Accuracy'][idx_min],
                                                                                    data['RMSE1'][idx_min],
                                                                                    data['RMSE2'][idx_min],
                                                                                    data['RMSE3'][idx_min],
                                                                                    data['RMSE'][idx_min],
                                                                                    data['MAE1'][idx_min],
                                                                                    data['MAE2'][idx_min],
                                                                                    data['MAE3'][idx_min],
                                                                                    data['MAE'][idx_min]))
        fw.close()

        # if cfg['updateJSON'] is True:
        #     cfg_updating(cfg, 'featuresSelection', 'FS', region, None)
        #     cfg_updating(cfg, 'modelTraining', 'MT', region, '%s-%s' % (cfg['outputTarget'], day_ahead))

    print('Ending program')
