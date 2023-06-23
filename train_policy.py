import os
from datetime import datetime
import gym
import torch
import random
import numpy as np

from PPO import PPO
from reps.MVTCAE import MVTCAE
from reps.MVSSM import MVSSM
from reps.CMC import CMC
from envs.MVBipedal import dropStateMVBipedalMean, genMVBipedal, dropStateMVBipedal


def train(config):
    env_name = config['env_name']
    method_name = config['method']
    is_MV = config['is_MV']
    missing_view_num = config['missing_view_num']

    max_ep_len = config['max_ep_len']
    max_training_timesteps = config['max_training_timesteps']
    print_freq = config['print_freq']
    log_freq = config['log_freq']
    save_model_freq = config['save_model_freq']
    action_std = config['action_std']
    action_std_decay_rate = config['action_std_decay_rate']
    min_action_std = config['min_action_std']
    action_std_decay_freq = config['action_std_decay_freq']

    rl_algorithm = config['rl_algorithm']

    has_continuous_action_space = True  # continuous action space; else discrete

    print("============================================================================================")
    trained_model_filename = '_'.join([str(config[k]) for k in config['save_keys_reps']]) + '_best.pth'

    ####### initialize environment hyperparameters ######
    if env_name in ['BipedalWalker-v3'] and method_name in ["MVTCAE", "MVSSM", "SLAC", "CMC", "Vanilla-RL"]:
        trained_model_dir = './representations/{}/{}_pretrained/'.format(method_name, rl_algorithm)
        trained_model_path = os.path.join(trained_model_dir, trained_model_filename)
    else:
        raise ValueError(f'Invalid environment name / method name: {env_name} / {method_name}')
    print(trained_model_path)
    ################ RL hyperparameters ################
    update_timestep = config['update_timestep']            # update policy every n timesteps
    K_epochs = config['K_epochs']                          # update policy for K epochs in one actor update
    gamma = config['gamma']                                # discount factor
    lr_actor = config['lr_actor']                          # learning rate for actor network
    lr_critic = config['lr_critic']                        # learning rate for critic network
    eps_clip = 0.2                                         # clip parameter for PPO

    ### Environment setup ###
    print("training environment name : " + env_name)

    if not env_name in ['BipedalWalker-v3']:
        raise ValueError(f'Invalid env name ', env_name)

    env = gym.make(env_name)

    torch.manual_seed(config['seed'])
    env.seed(config['seed'])
    np.random.seed(config['seed'])

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    genMV = genMVBipedal
    _, _, num_views, view_sizes, view_binaries = genMV(np.zeros((1, state_dim), dtype=np.float32))

    use_subset = [False] * num_views

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### create new log file for each run
    log_f_name = log_dir + 'PPO_' + env_name + '_rep_' + '_'.join([str(config[k]) for k in config['save_keys_train']]) + '_log.csv'

    print("current logging run number for " + env_name + " : ", config['seed'])
    print("logging at : " + log_f_name)
    #####################################################

    directory = f'rep_{rl_algorithm}_Trained'

    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + 'rep' + '_'.join([str(config[k]) for k in config['save_keys_train']]) + '.pth'
    best_checkpoint_path = directory + 'rep' + '_'.join([str(config[k]) for k in config['save_keys_train']]) + '_best.pth'

    print("save checkpoint path : " + checkpoint_path)

    if os.path.isfile(checkpoint_path):
        print('Trained policy already exists!!!')
        return

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)

    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    print("setting random seed to ", config['seed'])
    torch.manual_seed(config['seed'])
    env.seed(config['seed'])
    np.random.seed(config['seed'])
    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    if is_MV:
        if method_name in ['MVTCAE', 'MVSSM', 'SLAC', 'CMC']:
            if method_name == 'MVTCAE':
                # Load pretrained representation model
                representation = MVTCAE(config['rec_coeff'], config['activation'], config['n_latents'], view_sizes, 
                                        config['hidden_size'], view_binaries, use_prior_expert=False)
                representation.load_state_dict(torch.load(trained_model_path)['model_state_dict'])
                print('Pretrained MVTCAE is loaded.')

            elif method_name == 'CMC':
                representation = CMC(config['rec_coeff'], config['activation'], config['n_latents'], view_sizes, 
                                    config['hidden_size'], view_binaries, use_prior_expert=False)
                representation.load_state_dict(torch.load(trained_model_path)['model_state_dict'])
                print('Pretrained CMC is loaded.')

            elif method_name in ['MVSSM', 'SLAC']:
                representation = MVSSM(config['rec_coeff'], config['activation'], config['n_latents'], view_sizes, config['hidden_size'], 
                                       4, view_binaries, use_prior_expert=False, method=method_name)
                representation.load_state_dict(torch.load(trained_model_path)['model_state_dict'])
                print('Pretrained MVSSM is loaded.')
            else:
                raise ValueError(f'Invalid method name ', method_name)
            representation.eval()
            agent = PPO(config['n_latents'], action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        elif method_name == 'Vanilla-RL':
            # subset_dim = sum([int(v_s * v_u) for v_s, v_u in zip(view_sizes, use_subset)])
            agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        else:
            raise ValueError(f'Invalid method name ', method_name)
    else:
        agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    training_info = {
        'iteration': 0,
        'logs': [],
    }

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0
    best_score = -1000
    i_episode = 0

    def sample_random_subset(missing_view_num):
        sampled_subset = [True] * num_views
        view_indices = [i for i in range(num_views)]

        missing_view_indices = random.sample(view_indices, missing_view_num)
        for v in missing_view_indices:
            sampled_subset[v] = False
        return sampled_subset

    def sample_doubly_random_subset():
        sampled_subset = [True] * num_views
        view_indices = [i for i in range(num_views)]
        missing_view_num = random.sample([i for i in range(num_views)], 1)[0]

        missing_view_indices = random.sample(view_indices, missing_view_num)
        for v in missing_view_indices:
            sampled_subset[v] = False
        return sampled_subset

    def preprocess_state(state, action, use_prior, missing_view_num):
        if is_MV:
            views, _, _, _, _ = genMV(state)
            if method_name in ['MVTCAE', 'MVSSM', 'CMC', 'SLAC']:
                if missing_view_num == 0:
                    use_subset = [True] * num_views
                elif missing_view_num == num_views:
                    use_subset = sample_doubly_random_subset()
                elif missing_view_num != 0:
                    use_subset = sample_random_subset(missing_view_num)
                elif missing_view_num > num_views:
                    print('# of missing view > # of views')
                    raise ValueError

                if method_name == 'MVTCAE':
                    state, _, _, _, _ = representation.infer([torch.Tensor(v) for v in views],
                                                             use_prior_expert=False,
                                                             use_subset=use_subset)
                    state = state.detach().cpu().numpy().flatten()

                elif method_name == 'CMC':
                    state, _ = representation.infer([torch.Tensor(v) for v in views],
                                                    use_subset=use_subset)
                    state = state.detach().cpu().numpy().flatten()
                    
                else:
                    state, _, _, _, _, _, _ = representation.infer(
                        [torch.Tensor(np.reshape(v, (1, 1, -1))) for v in views],
                        action=torch.Tensor(np.reshape(action, (1, 1, -1))),
                        use_prior=use_prior,
                        use_subset=use_subset)

                    use_prior = state.view(1,-1).detach().cpu()
                    state = use_prior.numpy().flatten()
            elif method_name == 'Vanilla-RL':
                # TODO: Choose subset of views before concatenating all.
                if missing_view_num > 0:                    
                    if missing_view_num == num_views:
                        use_subset = sample_doubly_random_subset()
                        num_true_subset = (np.array(use_subset) == True).sum()
                        if num_true_subset != num_views:
                            if config['impute_style'] == 'zero':
                                state = dropStateMVBipedal(state, use_subset)
                            elif config['impute_style'] == 'mean':
                                state = dropStateMVBipedalMean(state, use_subset)
                            
                    else:
                        use_subset = sample_random_subset(missing_view_num)
                        
                        if config['impute_style'] == 'zero':
                            state = dropStateMVBipedal(state, use_subset)
                        elif config['impute_style'] == 'mean':
                            state = dropStateMVBipedalMean(state, use_subset)

                state = np.reshape(state, (-1,))
            else:
                raise ValueError(f'Invalid method name ', method_name)

        return state, use_prior

    # training loop
    while training_info['iteration'] <= max_training_timesteps:
        use_prior = None
        state = env.reset()
        action = np.zeros((action_dim,), dtype=np.float32)
        state, use_prior = preprocess_state(state, action, use_prior, missing_view_num)
        current_ep_reward = 0
        for t in range(1, max_ep_len+1):
            action = agent.select_action(state)

            state, reward, done, _ = env.step(action)
            state, use_prior = preprocess_state(state, action, use_prior, missing_view_num)

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            training_info['iteration'] +=1
            current_ep_reward += reward

            # update PPO agent
            if training_info['iteration'] % update_timestep == 0:
                agent.update()

            if has_continuous_action_space and training_info['iteration'] % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if training_info['iteration']  % log_freq == 0:

                # log average reward till last episode
                # Note: this value is used for evaluation.
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                info_dict = {'train_log_avg_reward': log_avg_reward}

                log_f.write('{},{},{}\n'.format(i_episode, training_info['iteration'], log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
                
                if best_score < log_avg_reward:
                        agent.save(best_checkpoint_path)
                        best_score = log_avg_reward

                if best_score < log_avg_reward:
                    agent.save(best_checkpoint_path)
                    best_score = log_avg_reward

            # printing average reward
            if training_info['iteration'] % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                current_time = datetime.now().replace(microsecond=0)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Time: {}".format(i_episode, training_info['iteration'], print_avg_reward, current_time - start_time))

                print_running_reward = 0
                print_running_episodes = 0

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    agent.save(checkpoint_path)

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
