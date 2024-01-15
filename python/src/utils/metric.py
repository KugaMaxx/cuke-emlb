from datetime import timedelta
from typing import Tuple, List

# Import dv related package
import dv_processing as dv
import dv_toolkit as kit

# Import numpy
import numpy as np
from numpy.lib.stride_tricks import as_strided


def median_filter(data, size=3):
    # use following package instead can accelerate
    # from scipy.ndimage import median_filter
    temp = np.pad(data, [(size//2, size//2)], mode='constant')
    view = as_strided(temp, shape=(data.shape[0], data.shape[1], size, size), 
                      strides=(temp.strides[0], temp.strides[1], temp.strides[0], temp.strides[1]))
    filtered = np.median(view, axis=(2, 3))

    return filtered


class EventStructuralRatioV2(object):
    def __init__(self, resolution) -> None:
        self.resolution = resolution
        self.accumulator = dv.Accumulator(resolution)

        # set accumulator
        self.accumulator.setMinPotential(-np.inf)
        self.accumulator.setMaxPotential(np.inf)
        self.accumulator.setEventContribution(1.0)
        self.accumulator.setIgnorePolarity(True)
        self.accumulator.setDecayFunction(dv.Accumulator.Decay.NONE)
    
    def evalPerTimeInterval(self, data, 
                            reference: str = "events",
                            interval: timedelta = timedelta(milliseconds=33)):
        # slice data
        slicer, score = kit.MonoCameraSlicer(), list()
        slicer.doEveryTimeInterval(
            reference, interval, 
            lambda data: score.append(self._calc_esr(data["events"].toEventStore()))
        )
        slicer.accept(data)

        # return result
        return np.array(score)

    def evalPerNumberInterval(self, data, 
                              reference: str = "events",
                              interval: int = 30000):
        # slice data
        slicer, score = kit.MonoCameraSlicer(), list()
        slicer.doEveryNumberOfElements(
            reference, interval, 
            lambda data: score.append(self._calc_esr(data["events"].toEventStore()))
        )
        slicer.accept(data)
        
        # return result
        return np.array(score)
    
    def evalEventStorePerTime(self, events: dv.EventStore, 
                              interval: timedelta = timedelta(milliseconds=33)):
        # slice data
        slicer, score = dv.EventStreamSlicer(), list()
        slicer.doEveryTimeInterval(
            interval, 
            lambda events: score.append(self._calc_esr(events))
        )
        slicer.accept(events)
        
        # return result
        return np.array(score)

    def evalEventStorePerNumber(self, events: dv.EventStore, 
                                interval: int = 30000):
        # slice data
        slicer, score = dv.EventStreamSlicer(), list()
        slicer.doEveryNumberOfEvents(
            interval, 
            lambda events: score.append(self._calc_esr(events))
        )
        slicer.accept(events)
        
        # return result
        return np.array(score)

    def _calc_esr(self, events):
        self.accumulator.clear()
        self.accumulator.accept(events)
        
        # get basic info
        n = median_filter(self.accumulator.getPotentialSurface(), size=3)
        N = events.size()
        K = self.resolution[0] * self.resolution[1]  # n.size

        # calculate ntss
        ntss = (n * n).sum() / (N * N)
        
        # calculate ln
        ln = (K - (0.5 ** n).sum()) / K

        # return esr
        return 1000 * np.sqrt(ntss * ln)


class EventStructuralRatio(object):
    def __init__(self, resolution) -> None:
        self.resolution = resolution
        self.accumulator = dv.Accumulator(resolution)

        # set accumulator
        self.accumulator.setMinPotential(-np.inf)
        self.accumulator.setMaxPotential(np.inf)
        self.accumulator.setEventContribution(1.0)
        self.accumulator.setIgnorePolarity(True)
        self.accumulator.setDecayFunction(dv.Accumulator.Decay.NONE)
    
    def evalPerTimeInterval(self, data, 
                            reference: str = "events",
                            interval: timedelta = timedelta(milliseconds=33)):
        # slice data
        slicer, score = kit.MonoCameraSlicer(), list()
        slicer.doEveryTimeInterval(
            reference, interval, 
            lambda data: score.append(self._calc_esr(data["events"].toEventStore()))
        )
        slicer.accept(data)

        # return result
        return np.array(score)

    def evalPerNumberInterval(self, data, 
                              reference: str = "events",
                              interval: int = 30000):
        # slice data
        slicer, score = kit.MonoCameraSlicer(), list()
        slicer.doEveryNumberOfElements(
            reference, interval, 
            lambda data: score.append(self._calc_esr(data["events"].toEventStore()))
        )
        slicer.accept(data)
        
        # return result
        return np.array(score)
    
    def evalEventStorePerTime(self, events: dv.EventStore, 
                              interval: timedelta = timedelta(milliseconds=33)):
        # slice data
        slicer, score = dv.EventStreamSlicer(), list()
        slicer.doEveryTimeInterval(
            interval, 
            lambda events: score.append(self._calc_esr(events))
        )
        slicer.accept(events)
        
        # return result
        return np.array(score)

    def evalEventStorePerNumber(self, events: dv.EventStore, 
                                interval: int = 30000):
        # slice data
        slicer, score = dv.EventStreamSlicer(), list()
        slicer.doEveryNumberOfEvents(
            interval, 
            lambda events: score.append(self._calc_esr(events))
        )
        slicer.accept(events)
        
        # return result
        return np.array(score)

    def _calc_esr(self, events):
        self.accumulator.clear()
        self.accumulator.accept(events)
        
        # get basic info
        n = self.accumulator.getPotentialSurface()
        N, M = events.size(), int(events.size() * 2 / 3)
        K = self.resolution[0] * self.resolution[1]  # n.size

        # calculate ntss
        ntss = (n * (n - 1)).sum() / (N + np.spacing(1)) / (N - 1 + np.spacing(1))
        
        # calculate ln
        ln = K - ((1 - M / N) ** n).sum()

        # return esr
        return np.sqrt(ntss * ln)
