import logging
import pickle
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from envs.MVBipedal import MVBipedalDataset, MVBipedalSeqDataset

from reps.MVTCAE import MVTCAE
from reps.CMC import CMC
from reps.MVSSM import MVSSM
from reps.trainer_tester import trainer, tester


def train(config):
    env_name = config['env_name']
    method_name = config['method']
    cuda = config['cuda']

    if env_name in ['BipedalWalker-v3'] and method_name in ["MVTCAE", "MVSSM", "SLAC", "CMC"]:
        root_dir = './'
        log_dir = f'representations/{method_name}/'
    else:
        raise ValueError(f'Invalid environment name / method name: {env_name} / {method_name}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    trained_model_dir = os.path.join(log_dir, '{}_pretrained'.format(config['rl_algorithm']))
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    should_resume = False

    trained_model_filename = '_'.join([str(config[k]) for k in config['save_keys_reps']]) + '.pth'
    best_trained_model_filename = '_'.join([str(config[k]) for k in config['save_keys_reps']]) + '_best.pth'
    
    trained_model_path = os.path.join(trained_model_dir, trained_model_filename)
    best_trained_model_path = os.path.join(trained_model_dir, best_trained_model_filename)
    
    if os.path.isfile(trained_model_path):
        checkpoint = torch.load(trained_model_path)
        if checkpoint['epoch'] == config['epochs']:
            print('Fully trained Representation already exists!!!')
            return
        else:
            print('Resuming Representation Pretraining!!')
            should_resume = True
    else:
        print('Pretraining Representation from the scratch!')
    
    train_data_filename = 'dataset/pretrain_train_data_PPO.pkl'
    train_data_dir = root_dir + train_data_filename

    valid_data_filename = 'dataset/pretrain_valid_data_PPO.pkl'
    valid_data_dir = root_dir + valid_data_filename    

    ### Load precollected trajectories & Generate multi-view data ##
    if env_name in ['BipedalWalker-v3']:
        logging.info('Reading `{}` file...'.format(train_data_dir))
        with open(train_data_dir, 'rb') as file:
            input_train = pickle.load(file)

        logging.info('Reading `{}` file...'.format(valid_data_dir))
        with open(valid_data_dir, 'rb') as file:
            input_valid = pickle.load(file)

        if method_name in ['MVTCAE', 'CMC']:
            MVBipedal_train = MVBipedalDataset(input_train, cuda)
            MVBipedal_valid = MVBipedalDataset(input_valid, cuda)
            action_size = None
        elif method_name in ['MVSSM', 'SLAC']:
            MVBipedal_train = MVBipedalSeqDataset(input_train, 8, cuda)
            MVBipedal_valid = MVBipedalSeqDataset(input_valid, 8, cuda)
            action_size = MVBipedal_train.get_action_size()
        else:
            raise ValueError(f'Invalid method name ', method_name)

        view_sizes = MVBipedal_train.get_view_sizes()
        view_binaries = MVBipedal_train.get_view_binaries()
        train_loader = DataLoader(MVBipedal_train, batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=0, drop_last=True)
        valid_loader = DataLoader(MVBipedal_valid, batch_size=config['batch_size'],
                                  shuffle=False,
                                  num_workers=0, drop_last=True)

        print("Num train : ", MVBipedal_train.num_samples)
        print("Num valid : ", MVBipedal_valid.num_samples)
    else:
        raise ValueError(f'Invalid environment name: {env_name}')

    if config['even_dim_recon']:
        max_view_size = max(view_sizes)
        config['lambda_views'] = [max_view_size / float(view_size) for view_size in view_sizes]
    else:
        config['lambda_views'] = [1.0]*len(view_sizes)

    N_mini_batches = len(train_loader)

    ### Create a model ###
    if method_name == 'MVTCAE':
        config['use_prior_expert'] = False
        model = MVTCAE(config['rec_coeff'], config['activation'], config['n_latents'], view_sizes, config['hidden_size'],
                       view_binaries, use_prior_expert=config['use_prior_expert'])
    elif method_name == 'CMC':
        model = CMC(config['rec_coeff'], config['activation'], config['n_latents'], view_sizes, config['hidden_size'],
                    view_binaries, use_prior_expert=config['use_prior_expert'])
    elif method_name in ['MVSSM', 'SLAC']:
        config['use_prior_expert'] = False
        model = MVSSM(config['rec_coeff'], config['activation'], config['n_latents'], view_sizes, config['hidden_size'], 
                      action_size, view_binaries, use_prior_expert=config['use_prior_expert'], method=method_name)
    else:
        raise ValueError(f'Invalid method name: {method_name}')

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    start_epoch = 0

    training_info = {
        'epoch': 0,
        'logs': [],
    }

    min_valid_loss = 100000.
    if should_resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        min_valid_loss = checkpoint['min_valid_loss']
        print('-- resumed training information -- ')
        print(' ** start_epoch : ', start_epoch)
        print(' ** min_valid_loss : ', min_valid_loss)

    if config['cuda']:
        model.cuda()

    for training_info['epoch'] in range(start_epoch+1, config['epochs'] + 1):
        loss_dict = trainer(training_info['epoch'], model, train_loader, config['cuda'], optimizer,
                            config['lambda_views'], config['anneal_epochs'], N_mini_batches, config['log_interval'])        

        _, avg_valid_loss = tester(training_info['epoch'], model, valid_loader, config['lambda_views'])

        print(f'---- Save latest model ----')
        torch.save({
            'epoch': training_info['epoch'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'min_valid_loss': min_valid_loss
        }, trained_model_path)

        if min_valid_loss >= avg_valid_loss:
            min_valid_loss = avg_valid_loss
            print(f'---- Save best model (valid loss : ', min_valid_loss, ' )')
            torch.save({
                'epoch': training_info['epoch'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'min_valid_loss': min_valid_loss
            }, best_trained_model_path)
