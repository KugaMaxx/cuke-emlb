import os
from typing import Dict
import os.path as osp
import numpy as np
from abc import ABC, abstractmethod
from datetime import timedelta

import dv_toolkit as kit

    
class Module(ABC):
    def __init__(self, model, resolution, 
                 modified_params: Dict, 
                 default_params: Dict) -> None:
        super().__init__()

        # overwrite default parameters
        self.params = default_params
        for key, value in modified_params.items():
            self.params[key] = value

        # initialize model
        self.model = model.init(resolution, **self.params)

    def accept(self, events):
        self.model.accept(events)
    
    def generateEvents(self):
        return self.model.generateEvents()


class dwf(Module):
    def __init__(self, resolution, modified_params, 
                 default_params = {"bufferSize": 36,
                                   "searchRadius": 9,
                                   "intThreshold": 1}):
        from modules.python import double_window_filter as module
        super().__init__(module, resolution, modified_params, default_params)


class edncnn(Module):
    def __init__(self, resolution, modified_params, 
                 default_params = {"batchSize": 1000,
                                   "floatThreshold":0.5,
                                   "deviceId": 0}):
        default_params["modelPath"] = osp.join(osp.dirname(__file__), 
                                               '../modules/net/EDnCNN_all_trained_v9.pt')
        from modules.python import event_denoise_convolution_network as module
        super().__init__(module, resolution, modified_params, default_params)


class evflow(Module):
    def __init__(self, resolution, modified_params, 
                 default_params = {"duration": timedelta(milliseconds=2),
                                   "searchRadius": 1,
                                   "floatThreshold": 20.0}):
        from modules.python import event_flow as module
        super().__init__(module, resolution, modified_params, default_params)


class knoise(Module):
    def __init__(self, resolution, modified_params, 
                 default_params = {"duration": timedelta(milliseconds=2),
                                   "intThreshold": 1}):
        from modules.python import khodamoradi_noise as module
        super().__init__(module, resolution, modified_params, default_params)


class mlpf(Module):
    def __init__(self, resolution, modified_params, 
                 default_params = {"batchSize": 10000000,
                                   "duration": timedelta(milliseconds=100),
                                   "floatThreshold":0.8,
                                   "deviceId": 0}):
        default_params["modelPath"] = osp.join(osp.dirname(__file__), 
                                               '../modules/net/MLPF_2xMSEO1H20_linear_7.pt')
        from modules.python import multi_layer_perceptron_filter as module
        super().__init__(module, resolution, modified_params, default_params)


class red(Module):
    def __init__(self, resolution, modified_params, 
                 default_params = {"sigmaS": 0.7,
                                   "sigmaT": 1,
                                   "samplarT": -0.8,
                                   "floatThreshold": 0.25}):
        from modules.python import reclusive_event_denoisor as module
        super().__init__(module, resolution, modified_params, default_params)


class ts(Module):
    def __init__(self, resolution, modified_params, 
                 default_params = {"duration": timedelta(milliseconds=2),
                                   "searchRadius": 1,
                                   "floatThreshold": 0.2}):
        from modules.python import time_surface as module
        super().__init__(module, resolution, modified_params, default_params)


class ynoise(Module):
    def __init__(self, resolution, modified_params, 
                 default_params = {"duration": timedelta(milliseconds=2),
                                   "searchRadius": 1,
                                   "intThreshold": 1}):
        from modules.python import yang_noise as module
        super().__init__(module, resolution, modified_params, default_params)


# class evzoom(BaseDenoisor):
#     def __init__(self, params={}):
#         super().__init__(
#             "TODO",
#             **params
#         )


def Denoisor(model_name, resolution, modified_params: Dict = dict()):
    # abbreviation mapping
    model_map = {
        "double_window_filter": "dwf",
        "event_denoise_convolution_network": "edncnn",
        "event_flow": "evflow",
        "khodamoradi_noise": "knoise",
        "multiLayer_perceptron_filter": "mlpf",
        "reclusive_event_denoisor": "red",
        "time_surface": "ts",
        "yang_noise": "ynoise",
    }

    # check supported list
    if model_name in model_map.keys():
        model_name = model_map[model_name]
    elif model_name in model_map.values():
        model_name = model_name
    else:
        raise KeyError("unsupported denoisors")

    # convert model string to model class
    model = eval(model_name)

    # send params to model
    return model(resolution, modified_params)
