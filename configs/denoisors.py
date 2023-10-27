import os
import os.path as osp
import numpy as np
from abc import ABC, abstractmethod

from modules import kore
from evtool.dtype import Event, Frame, Size


class BaseDenoisor(ABC):
    def __init__(self, model, **params):
        super().__init__()
        self.model = model
        self.params = params

    def run(self, data):
        model = self.model.init(data['size'][0], data['size'][1])
        idx = model.run(data['events'], **self.params)
        data['events'] = data['events'][idx]
        return data


class red(BaseDenoisor):
    def __init__(self, 
                 params={'samplarT': -1.5, 
                         'sigmaS': 0.3,
                         'sigmaT': 5, 
                         'threshold': 0.04}):
        from modules import reclusive_event_denoisor
        super().__init__(
            reclusive_event_denoisor,
            **params
        )


class ynoise(BaseDenoisor):
    def __init__(self, 
                 params={"delta_t": 10000, 
                         "square_r": 1,
                         "threshold": 2}):
        from modules import yang_noise
        super().__init__(
            yang_noise,
            **params
        )
    

class dwf(BaseDenoisor):
    def __init__(self, 
                 params={"w_len": 36,
                         "square_r": 9,
                         "threshold": 1}):
        from modules import double_window_filter
        super().__init__(
            double_window_filter,
            **params
        )


class knoise(BaseDenoisor):
    def __init__(self, 
                 params={"delta_t": 1000,
                         "threshold": 1}):
        from modules import khodamoradi_noise
        super().__init__(
            khodamoradi_noise,
            **params
        )


class ts(BaseDenoisor):
    def __init__(self, 
                 params={"decay": 30000,
                         "square_r": 1,
                         "threshold": 0.3}):
        from modules import time_surface
        super().__init__(
            time_surface,
            **params
        )


class evflow(BaseDenoisor):
    def __init__(self, 
                 params={"delta_t":3000,
                         "square_r": 1,
                         "threshold":2}):
        from modules import event_flow
        super().__init__(
            event_flow,
            **params
        )


class mlpf(BaseDenoisor):
    def __init__(self, 
                 params={"model_path": os.getcwd() + '/modules/_net/MLPF_2xMSEO1H20_linear_7.pt',
                         "batch_size": 10000000,
                         "decay": 100000,
                         "square_r": 3,
                         "threshold":0.8}):
        from modules import multiLayer_perceptron_filter
        super().__init__(
            multiLayer_perceptron_filter,
            **params
        )


class edncnn(BaseDenoisor):
    def __init__(self,
                 params={"model_path": os.getcwd() + '/modules/_net/EDnCNN_all_trained_v9.pt',
                         "batch_size": 1000,
                         "depth": 2,
                         "square_r": 12,
                         "threshold": 0.5}):
        from modules import event_denoise_convolution_network
        super().__init__(
            event_denoise_convolution_network,
            **params
        )


class evzoom(BaseDenoisor):
    def __init__(self, params={}):
        super().__init__(
            "TODO",
            **params
        )


def Denoisor(model_name):
    model = eval(model_name)
    return model()
