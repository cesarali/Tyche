import numpy as np


class ExponentialScheduler(object):

    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.decay_rate = kwargs.get('decay_rate', 0.1)

    def __call__(self, step):
        return float(1. / (1. + np.exp(-self.decay_rate * (step - self.max_steps))))


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
