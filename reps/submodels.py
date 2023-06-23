import torch
import torch.nn as nn
from torch.nn import functional as F

class IVWAvg(nn.Module):

    def forward(self, mu, logvar):
        var = torch.exp(logvar)
        inv_var = 1. / var

        joint_mu = torch.sum(mu * inv_var, dim=0) / torch.sum(inv_var, dim=0)
        joint_var = 1. / torch.sum(inv_var, dim=0)
        joint_logvar = torch.log(joint_var)
        return joint_mu, joint_logvar


class AttributeEncoder(nn.Module):

    def __init__(self, activation, n_latents, in_size, hidden_size=64):
        super(AttributeEncoder, self).__init__()
        if activation == 'leakyrelu':
            self.net = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_size, n_latents * 2))
        else:
            raise ValueError(f'Invalid activation ', activation)
        self.n_latents = n_latents

    def forward(self, x):
        h = self.net(x)
        return h[:, :self.n_latents], h[:, self.n_latents:]


class AttributeDecoder(nn.Module):

    def __init__(self, activation, n_latents, out_size, hidden_size=64, is_binary=False):
        super(AttributeDecoder, self).__init__()
        if activation == 'leakyrelu':
            net = nn.Sequential(
                nn.Linear(n_latents, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_size, out_size))
        else:
            raise ValueError(f'Invalid activation', activation)
        if is_binary:
            print("Binary")
            self.net = nn.Sequential(net, Sigmoid())
        else:
            self.net = net

    def forward(self, z):
        x = self.net(z)
        return x


class DetEncoder(nn.Module):

    def __init__(self, activation, n_latents, in_size, hidden_size=64):
        super(DetEncoder, self).__init__()
        if activation == 'tanh':
            self.net = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, n_latents))
        elif activation == 'leakyrelu':
            self.net = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_size, n_latents))
        elif activation == 'small_leakyrelu':
            self.net = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_size, n_latents))
        else:
            raise ValueError(f'Invalid activation ', activation)
        self.n_latents = n_latents

    def forward(self, x):
        h = self.net(x)
        return h #[:, :self.n_latents], h[:, self.n_latents:]
    

class Swish(nn.Module): # https://arxiv.org/abs/1710.05941
    def forward(self, x):
        return x * torch.sigmoid(x)


class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)
