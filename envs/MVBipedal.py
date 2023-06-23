import numpy as np
import torch
import random
import logging

from torch.utils.data import Dataset
from torch.autograd import Variable


# # valid
# state_mean = np.array([ 0.1105869 , -0.00114103,  0.40424709, -0.0037641 , -0.08847108,
#         0.01721298, -0.30858639, -0.08151781,  0.47898277,  1.0990507 ,
#         0.03671096,  0.41065881, -0.01532206,  0.22140897,  0.333168  ,
#         0.33689326,  0.34863813,  0.36986741,  0.40354676,  0.4552791 ,
#         0.53611845,  0.67027975,  0.90429936,  0.99948804])

# # train
state_mean = np.array([ 0.11585174, -0.00111921,  0.38659506, -0.00396042, -0.12335914,
        0.0156904 , -0.30298461, -0.07947137,  0.48079713,  1.09705567,
        0.03542111,  0.42500328, -0.01111571,  0.24357939,  0.32779013,
        0.3314497 ,  0.34300172,  0.36388281,  0.39700565,  0.4478885 ,
        0.52741063,  0.65932857,  0.88952934,  0.99898717])

logging.basicConfig(level=logging.INFO)


def genMVBipedal(states):
    if len(states.shape) == 1:
        states = np.reshape(states, (1, -1))

    views = [states[:, [0, 4, 6, 9, 11]],   # Angles
             states[:, [1, 5, 7, 10, 12]],  # Angular speed
             states[:, [2, 3]],             # Horizontal / Vertical speed
             states[:, [8, 13]],            # Ground contact (Binary)
             states[:, 14:]]                # Lidar measurements

    view_binaries = [False, False, False, True, False]
    num_samples = states.shape[0]
    num_views = len(views)
    view_sizes = [v.shape[-1] for v in views]
    return views, num_samples, num_views, view_sizes, view_binaries

def dropStateMVBipedal(states, sampled_subset):
    if len(states.shape) == 1:
        states = np.reshape(states, (1, -1))

    state_idx = np.arange(states.shape[1])
    state_idx_per_view = [state_idx[[0, 4, 6, 9, 11]],   # Angles
                          state_idx[[1, 5, 7, 10, 12]],  # Angular speed
                          state_idx[[2, 3]],             # Horizontal / Vertical speed
                          state_idx[[8, 13]],            # Ground contact (Binary)
                          state_idx[14:]]                # Lidar measurements

    drop_view_idx = np.where(np.logical_not(sampled_subset))[0]
    drop_state_idx_total = np.concatenate(np.array(state_idx_per_view)[drop_view_idx]) 
    drop_state_idx_total.sort()    
    states[:, drop_state_idx_total] = 0.
    return states

def dropStateMVBipedalMean(states, sampled_subset):
    if len(states.shape) == 1:
        states = np.reshape(states, (1, -1))

    state_idx = np.arange(states.shape[1])
    state_idx_per_view = [state_idx[[0, 4, 6, 9, 11]],   # Angles
                          state_idx[[1, 5, 7, 10, 12]],  # Angular speed
                          state_idx[[2, 3]],             # Horizontal / Vertical speed
                          state_idx[[8, 13]],            # Ground contact (Binary)
                          state_idx[14:]]                # Lidar measurements

    drop_view_idx = np.where(np.logical_not(sampled_subset))[0]
    drop_state_idx_total = np.concatenate(np.array(state_idx_per_view)[drop_view_idx]) 
    drop_state_idx_total.sort()    
    states[:, drop_state_idx_total] = state_mean[drop_state_idx_total]
    return states


class MVBipedalDataset(Dataset):
    def __init__(self, input_data, is_cuda):
        super().__init__()
        states = np.array(input_data.states, dtype=np.float32)
        actions= np.array(input_data.actions, dtype=np.float32)
        views, num_samples, num_views, view_sizes, view_binaries = genMVBipedal(states)
        action_size = actions.shape[-1]

        self.num_views = num_views

        self.views = [torch.Tensor(v) for v in views]
        self.actions = torch.Tensor(actions)

        self.view_binaries = view_binaries

        self.num_samples = num_samples
        self.view_sizes = view_sizes
        self.action_size = action_size
        self.is_cuda = is_cuda

    def get_view_sizes(self):
        return self.view_sizes

    def get_action_size(self):
        return self.action_size

    def get_view_binaries(self):
        return self.view_binaries

    def __getitem__(self, index):
        if self.is_cuda:
            return [Variable(self.views[i][index].cuda()) for i in range(self.num_views)], Variable(self.actions[index].cuda())
        else:
            return [Variable(self.views[i][index]) for i in range(self.num_views)], Variable(self.actions[index])

    def __len__(self):
        return self.num_samples


class MVBipedalSeqDataset(Dataset):
    def __init__(self, input_data, seq_len, is_cuda):
        super().__init__()
        states = np.array(input_data.states, dtype=np.float32)
        actions= np.array(input_data.actions, dtype=np.float32)
        views, num_samples, num_views, view_sizes, view_binaries = genMVBipedal(states)
        action_size = actions.shape[-1]

        self.num_views = num_views

        self.views = [torch.Tensor(v) for v in views]
        self.actions=torch.Tensor(actions)
        self.init_traj_indices = input_data.init_traj_indices

        self.avoid_indices = []
        for idx in input_data.init_traj_indices:
            for ad in range(seq_len-1):
                self.avoid_indices.append(idx + ad)

        self.view_binaries = view_binaries

        self.num_samples = num_samples
        self.view_sizes = view_sizes
        self.action_size = action_size
        self.seq_len = seq_len
        self.is_cuda = is_cuda

    def get_view_sizes(self):
        return self.view_sizes

    def get_action_size(self):
        return self.action_size

    def get_view_binaries(self):
        return self.view_binaries

    def __getitem__(self, index):
        while index in self.avoid_indices:
            index = random.randint(0, self.num_samples-1)

        if self.is_cuda:
            return [Variable(self.views[i][index-(self.seq_len-1):index+1].cuda()) for i in range(self.num_views)], Variable(self.actions[index-(self.seq_len-1):index+1].cuda())
        else:
            return [Variable(self.views[i][index-(self.seq_len-1):index+1]) for i in range(self.num_views)], Variable(self.actions[index-(self.seq_len-1):index+1])

    def __len__(self):
        return self.num_samples
