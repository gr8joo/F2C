import argparse

import collect_data
import pretrain_reps
import train_policy

import os
# import wget
import importlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', default='BipedalWalker', help='environment name')
    parser.add_argument("--config", help="config filename", default='bipedal_config')
    parser.add_argument("--method", help="method name", default='MVSSM')                                            # Select MVSSM / SLAC / MVTCAE / Vanilla-RL
    parser.add_argument("--missing_view_num", help="# of missing views", default=0, type=int)                       # Select 0 / 1 / 2 / 3 / 4
    parser.add_argument("--seed", help="random seed id", default=0, type=int)                                       # Select 0 / 1 / 2 / 3 / 4
    parser.add_argument("--use_collected_data", help="skip the data collection process", default=True, type=bool)   # Select True / False
    
    args = parser.parse_args()
    config = vars(args)
    config_module = importlib.import_module('config.' + config['config'])
    env_hparams = config_module.env_hparams[0]

    for key in env_hparams.keys():
        config[key] = env_hparams[key]

    print(env_hparams)
    print('Method:', config['method'])
    if config['method'] in ['MVSSM', 'SLAC', 'MVTCAE', 'CMC']:
        ### 0. Data collection ###
        #     ### Only one time for initial ###
        if config['use_collected_data']:
            dataset_path = './dataset/'
            train_file_name = 'pretrain_train_data_PPO.pkl'
            valid_file_name = 'pretrain_valid_data_PPO.pkl'

            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            # Download pre-collected train dataset
            if not os.path.isfile(os.path.join(dataset_path, train_file_name)):
                print('DOWNLOAD PRE-COLLECTED TRAIN DATASET')
                url = 'https://zenodo.org/record/6583263/files/pretrain_train_data_PPO.pkl'
                wget.download(url, out=dataset_path)

            # Download pre-collected valid dataset
            if not os.path.isfile(os.path.join(dataset_path, valid_file_name)):
                print('DOWNLOAD PRE-COLLECTED VALID DATASET')
                url = 'https://zenodo.org/record/6583291/files/pretrain_valid_data_PPO.pkl'
                wget.download(url, out=dataset_path)
        else:
            collect_data.train(config)

        ### 1. Representation learning ###
        pretrain_reps.train(config)

        ### 2. Policy learning ###
        train_policy.train(config)

    elif config['method'] == 'Vanilla-RL':
        train_policy.train(config)

    else:
        raise ValueError("Incorrect method name ", config['method'])
