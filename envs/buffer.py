################################### Trajectories ###################################
class TrajBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.init_traj_indices = []
        self.total_num_samples = 0

    def add(self, state, action, reward, is_init=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

        if is_init:
            self.init_traj_indices.append(self.total_num_samples)

        self.total_num_samples +=1

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.init_traj_indices[:]