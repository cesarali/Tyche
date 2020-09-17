import numpy as np


class ExponentialScheduler(object):

    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.decay_rate = kwargs.get('decay_rate', 0.1)

    def __call__(self, step):
        return float(1. / (1. + np.exp(-self.decay_rate * (step - self.max_steps))))


class ExponentialIncrease(object):
    """
    Increases exponentially from zero to one
    """
    def __init__(self, **kwargs):
        n_steps_to_rich_maximum = kwargs.get('n_steps_to_rich_maximum', 10000)
        max_value = 0.99
        self.decay_rate = -np.log(1. - max_value) / n_steps_to_rich_maximum

    def __call__(self, step):
        return float(1. - np.exp(-self.decay_rate * step))


class ExponentialSchedulerGumbel(object):

    def __init__(self, **kwargs):
        self.min_tau = kwargs.get('min_temp')
        n_steps_to_rich_minimum = kwargs.get('n_steps_to_rich_minimum', 10000)
        self.decay_rate = -np.log(self.min_tau) / n_steps_to_rich_minimum

    def __call__(self, tau_init, step):
        t = np.maximum(tau_init * np.exp(-self.decay_rate * step), self.min_tau)
        return t


class LinearScheduler(object):
    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.start_value = kwargs.get('start_value', 0)
        print("start_value linear scheduler {}".format(self.start_value))

    def __call__(self, step):
        if self.start_value == 0:
            return min(1., float(step) / self.max_steps)
        else:
            return min(1., self.start_value + float(step) / self.max_steps * (1 - self.start_value))


class ConstantScheduler(object):
    def __init__(self, **kwargs):
        self.beta = kwargs.get('beta', 1.0)

    def __call__(self, step):
        return self.beta
