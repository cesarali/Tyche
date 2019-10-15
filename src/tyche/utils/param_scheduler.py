import numpy as np


class ExponentialScheduler(object):

    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.decay_rate = kwargs.get('decay_rate', 0.1)

    def __call__(self, step):
        return float(1. / (1. + np.exp(-self.decay_rate * (step - self.max_steps))))


class ExponentialSchedulerGumbel(object):

    def __init__(self, **kwargs):
        self.min_tau = kwargs.get('min_temp')
        self.decay_rate = kwargs.get('decay_rate')

    def __call__(self, tau_init, step):
        t = np.maximum(tau_init * np.exp(-self.decay_rate * step), self.min_tau)
        return t


class LinearScheduler(object):
    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)

    def __call__(self, step):
        return min(1., float(step) / self.max_steps)


class ConstantScheduler(object):
    def __init__(self, **kwargs):
        self.beta = kwargs.get('beta', 1000)

    def __call__(self, step):
        return self.beta
