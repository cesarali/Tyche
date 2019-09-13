# coding: utf-8

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import copy
import itertools
import json
from importlib import import_module

import numpy as np
import torch as to
import torch as torch
from scipy import linalg as la


def create_class_instance(module_name, class_name, kwargs, *args):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param class_name:
    :param kwargs: arguments needed for the class constructor
    :returns: instance of 'class_name'

    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    instance = clazz(*args, **kwargs)
    return instance


def create_instance(name, params, *args):
    """Creates an instance of class given configuration.

    :param name: of the module we want to create
    :param params: dictionary containing information how to instanciate the class
    :returns: instance of a class
    :rtype:

    """
    i_params = params[name]
    if type(i_params) is list:
        instance = [create_class_instance(
                p['module'], p['name'], p['args'], *args) for p in i_params]
    else:
        instance = create_class_instance(
                i_params['module'], i_params['name'], i_params['args'], *args)
    return instance


def create_nonlinearity(name):
    """
    Returns instance of nonlinearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz()

    return instance


def get_class_nonlinearity(name):
    """
    Returns nonlinearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)

    return clazz


def load_params(path):
    """Loads experiment parameters from json file.

    :param path: to the json file
    :returns: param needed for the experiment
    :rtype: dictionary

    """
    try:
        with open(path, "rb") as f:
            params = json.load(f)
        return params
    except:
        with open(path, "r") as f:
            params = json.load(f, encoding='utf-8')
        return params


def to_one_hot(labels, num_classes):
    """
    Convert tensor of labels to one hot encoding of the labels.
    :param labels: to be encoded
    :param num_classes:
    :return:
    """
    shape = labels.size()
    shape = shape + (num_classes,)
    one_hot = torch.FloatTensor(shape)
    one_hot.zero_()
    dim = 1 if len(shape) == 2 else 2
    one_hot.scatter_(dim, labels.unsqueeze(-1), 1)
    return one_hot


def expand_params(params):
    """
    Expand the hyperparamers for grid search

    :param params:
    :return:
    """
    cv_params = []
    if 'grid_search' in params:
        param_pool = [[(key, v) for v in value] for key,
                                                    value in params['grid_search'].items() if isinstance(value, list)]

        for i in list(itertools.product(*param_pool)):
            d = copy.deepcopy(params)
            name = d['name']
            for j in i:
                d[j[0]] = j[1]
                name += "_" + j[0] + "_" + str(j[1])
                d['name'] = name.replace('.args.', "_")
            cv_params.append(d)
    if not cv_params:
        return [params] * params['num_runs']

    gs_params = []
    for p in cv_params:
        gs_params += [p] * p['num_runs']
    return gs_params


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def gauss_legender_points(N=30):
    """ Returns the quadratures_nodes anb weights of the Gaussian-Lenegendre quadrature """
    beta = np.array([(n + 1.0) / np.sqrt((2.0 * n + 1.0) * (2.0 * n + 3.0))
                     for n in range(N - 1)], dtype=np.float32)
    M = np.diag(beta, -1) + np.diag(beta, 1)
    nodes, V = la.eigh(M, overwrite_a=True, overwrite_b=True)
    weight = 2 * V[0, :] ** 2
    return nodes, weight


def quadratures(f, a=-1, b=1, n=30):
    """
    Performing Legendre-Gauss quadrature integral approximation.

    :param f:
    :param a:
    :param b:
    :param n:
    :return:
    """
    nodes, weights = gauss_legender_points(n)
    w = to.tensor(weights.reshape(1, 1, -1))
    nodes = to.tensor(nodes.reshape(1, 1, -1))

    scale = (b - a) / 2.

    x = scale * nodes + (b + a) / 2.
    y = w * f(x)
    y = to.sum(scale * y, dim=-1)
    return y.type(dtype=to.float)


def gumbel_sample(shape, epsilon=1e-20):
    """
    Sample Gumbel(0,1)
    """
    u = get_cuda(torch.rand(shape))
    return -torch.log(-torch.log(u + epsilon) + epsilon)


def gumbel_softmax_sample(pi, tau):
    """
    Sample Gumbel-softmax
    """
    y = torch.log(pi) + gumbel_sample(pi.size())
    return torch.nn.functional.softmax(y / tau, dim=-1)


def gumbel_softmax(pi, tau):
    """
    Gumbel-Softmax distribution.
    Implementation from https://github.com/ericjang/gumbel-softmax.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
    """
    y = gumbel_softmax_sample(pi, tau)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


