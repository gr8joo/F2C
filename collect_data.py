import pickle
import torch
import numpy as np
import gym
import os
from datetime import datetime

from PPO import PPO
from envs.buffer import TrajBuffer


def train(config):
    print("============================================================================================")
    save_data = True
    env_name = config['env_name']
    rl_algorithm = config['rl_algorithm']

    if env_name in ['BipedalWalker-v3']:
        data_save_path = f'./dataset/'
    else:
        raise ValueError("Incorrect environment name ", env_name)

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = config['max_ep_len']
    max_training_timesteps = config['max_training_timesteps']   # break training loop if timeteps > max_training_timesteps

    print_freq = config['print_freq']                           # print avg reward in the interval (in num timesteps)
    log_freq = config['log_freq']                               # log avg reward in the interval (in num timesteps)
    save_model_freq = config['save_model_freq']                 # save model frequency (in num timesteps)

    action_std = config['action_std']                           # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = config['action_std_decay_rate']     # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = config['min_action_std']                   # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = config['action_std_decay_freq']     # action_std decay frequency (in num timesteps)

    ################ RL hyperparameters ################
    update_timestep = config['update_timestep']                 # update policy every n timesteps
    K_epochs = config['K_epochs']                               # update policy for K epochs in one actor update
    gamma = config['gamma']                                     # discount factor
    lr_actor = config['lr_actor']                               # learning rate for actor network
    lr_critic = config['lr_critic']                             # learning rate for critic network
    eps_clip = 0.2                                              # clip parameter for PPO

    print("training environment name : " + env_name)

    if env_name in ['BipedalWalker-v3']:
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

    # Otherwise, the environment must be our synthetic env.
    else:
        env = None
        state_dim = None
        action_dim = None

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "collect_data_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + f'/{rl_algorithm}_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    directory = f'{rl_algorithm}_preTrained'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "{}_{}_{}_{}_{}_{}_{}_seed_{}.pth".\
        format(rl_algorithm, env_name, config['action_std'], config['min_action_std'], 
               config['max_training_timesteps'], config['action_std_decay_freq'], config['update_timestep'], config['seed'])
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

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
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # Initialize TrajBuffer
    collect_data = dict(
            train = TrajBuffer(),
            valid = TrajBuffer()
            )
    valid_traj_ratio = 0.15

    # initialize a RL agent
    if 'PPO' in rl_algorithm:
        agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    else:
        raise ValueError("Incorrect RL algorithm name ", rl_algorithm)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    i_episode = 0

    training_info = {
        'iteration': 0,
        'logs': [],
    }

    # training loop
    if np.random.sample() < valid_traj_ratio:
        train_or_valid = 'valid'
    else:
        train_or_valid = 'train'

    while training_info['iteration'] <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            # select action with policy
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # Saving state, action, and rewards in TrajBuffer                        
            if save_data:
                if t>1:
                    collect_data[train_or_valid].add(state, action, reward)
                else:
                    collect_data[train_or_valid].add(state, action, reward, is_init=True)

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            training_info['iteration'] +=1
            current_ep_reward += reward

            # update agent
            if training_info['iteration'] % update_timestep == 0:
                agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and training_info['iteration'] % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if training_info['iteration'] % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                info_dict = {'collect_log_avg_reward': log_avg_reward}            

                log_f.write('{},{},{}\n'.format(i_episode, training_info['iteration'], log_avg_reward))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if training_info['iteration'] % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                current_time = datetime.now().replace(microsecond=0)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Time: {}".\
                    format(i_episode, training_info['iteration'], print_avg_reward, current_time-start_time))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if training_info['iteration'] % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                if np.random.sample() < valid_traj_ratio:
                    train_or_valid = 'valid'
                else:
                    train_or_valid = 'train'
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("Training buffer size : ",  collect_data["train"])
    print("Validation buffer size : ",  collect_data["valid"])
    print("============================================================================================")

    if save_data:
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)

        print("Saving training dataset..")
        trainfilename = 'pretrain_train_data_PPO.pkl'
        with open(os.path.join(data_save_path, trainfilename), 'wb') as save_collect_data:
            pickle.dump(collect_data["train"], save_collect_data)

        print("Saving valid dataset..")
        validfilename = 'pretrain_valid_data_PPO.pkl'
        with open(os.path.join(data_save_path, validfilename), 'wb') as save_collect_data:
            pickle.dump(collect_data["valid"], save_collect_data)
