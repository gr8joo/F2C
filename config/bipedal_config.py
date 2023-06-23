from itertools import product
from copy import deepcopy


grid_search = {
    "env_name": ["BipedalWalker-v3"],
                 
    # "method": ['MVSSM'],          # Select 'MVSSM', 'SLAC', 'MVTCAE', 'CMC', 'Vanilla-RL'
    # "missing_view_num": [0,1,2,3,4],     # 0: complete view, num_views: doubly random
    "batch_size": [512],
    "n_latents": [24],
    "hidden_size": [64],
    "epochs": [50],
    "lr": [3e-4],

    "anneal_epochs": [0],
    "log_interval": [100],
    "cuda": [False],
    "is_MV": [True],
    "activation": ["leakyrelu"],
    "rec_coeff": [5000.0],
    "impute_style": ['mean'],

    "save_model": [True],
    "even_dim_recon": [False],
    "use_prior_expert": [False],

    "max_ep_len": [1000],
    "max_training_timesteps": [int(3e6)],
    "print_freq": [10000],
    "log_freq": [2000],
    "save_model_freq": [2000],
    "action_std": [0.6],
    "action_std_decay_rate": [0.05],
    "min_action_std": [0.1],
    "action_std_decay_freq": [int(2.5e5)],

    "update_timestep": [4000],
    "K_epochs": [80],
    "lr_actor": [0.0003],
    "lr_critic": [0.001],
    "gamma": [0.99],

    "rl_algorithm": ["PPO"],

    "save_keys_reps":   [['method',
                          'action_std', 'min_action_std',
                          'max_training_timesteps', 'action_std_decay_freq',
                          'update_timestep', 'hidden_size', 'rec_coeff', 'activation', 'seed']],
    "save_keys_train":   [['rl_algorithm', 'env_name', 'method', 'action_std', 'min_action_std',
                           'missing_view_num',
                           'max_training_timesteps', 'action_std_decay_freq',
                           'update_timestep', 'hidden_size', 'rec_coeff', 'activation', 'seed']],

    "collect_data": [True],
    "pretrain_reps": [True],
    "train_policy": [True]
}

env_hparams = []

for grid_search_value in product(*grid_search.values()):
    grid_dict = dict(zip(grid_search.keys(), grid_search_value))

    hparam = deepcopy(grid_dict)
    env_hparams.append(hparam)

if __name__ == '__main__':
    for pid, hparam in enumerate(env_hparams):
        print(f'{pid:3d}: {hparam}')

