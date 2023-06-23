import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def reparameterize(mu, logvar, is_training):
    if is_training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu

def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

def bce_loss(input, target):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return torch.sum(F.binary_cross_entropy(input, target, reduction='none'), dim=-1)

def l1_loss(input, target):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return torch.sum(torch.abs(input - target), dim=-1)

def l2_loss(input, target):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return torch.sum((input - target)**2, dim=-1)

def kl_loss(mu1, logvar1, mu2=None, logvar2=None):
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    if mu2 is None:
        KLD = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=-1)
    else:
        KLD = -0.5 * torch.sum(1 + logvar1 - logvar2 - (mu1 - mu2).pow(2) / logvar2.exp() - logvar1.exp() / logvar2.exp() , dim=-1)
    return KLD

def swap_views(v1, v2):
    temp = v1
    v1 = v2
    v2 = temp
    return v1, v2

def preprocess(data, op, is_fitted=False):
    inv_log1p = None
    inv_stand = None
    if op['log1p'] is not None:
        # data = np.log1p(data)
        data = op['log1p'](data)
        inv_log1p = np.expm1

    if op['stand'] is not None:
        # func = StandardScaler().fit(data)
        if is_fitted is False:
            op['stand'] = op['stand'].fit(data)
        data = op['stand'].transform(data)
        inv_stand = op['stand'].inverse_transform

    return data, {'log1p': op['log1p'], 'stand': op['stand']}, {'log1p': inv_log1p, 'stand': inv_stand}

def deprocess(data, inv_op):
    if inv_op['stand'] is not None:
        data = inv_op['stand'](data)

    if inv_op['log1p'] is not None:
        data = inv_op['log1p'](data)

    return data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TBLogger():
    def __init__(self, writer):
        self.writer = writer
        self.training_prefix = 'train'
        self.testing_prefix = 'test'
        self.best_prefix = 'BEST'
        self.step = 0

    def write_train_logs(self, logs):
        for k, key in enumerate( sorted(logs.keys()) ):
            self.writer.add_scalar(self.training_prefix + '/' + key,
                                   logs[key],
                                   self.step)
    def write_test_logs(self, logs):
        for k, key in enumerate( sorted(logs.keys()) ):
            self.writer.add_scalar(self.testing_prefix + '/' + key,
                                   logs[key],
                                   self.step)
    def write_best_logs(self, logs):
        for k, key in enumerate( sorted(logs.keys()) ):
            self.writer.add_scalar(self.best_prefix + '/' + key,
                                   logs[key],
                                   0)
    def update_step(self, step):
        self.step = step
