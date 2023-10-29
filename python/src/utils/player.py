import numpy as np
from datetime import timedelta
from typing import Tuple, List

# Import dv related package
import dv_processing as dv
import dv_toolkit as kit

# Import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import dv_toolkit player component
from dv_toolkit.plot.tools import mp_player, go_player
from dv_toolkit.plot.tools.func import _visualize_events


class Preset(object):
    def __init__(self, size, packets) -> None:
        self._size = size[::-1]             # resolution
        self._packets = packets             # data
        self._ticks = range(len(packets))   # ticks

        # colormap
        self._fr_cmap = plt.get_cmap('gray')
        self._ev_cmap = mcolors.LinearSegmentedColormap.from_list("evCmap", [(0.871, 0.286, 0.247, 1.),
                                                                             (0.000, 0.000, 0.000, 0.),
                                                                             (0.180, 0.400, 0.600, 1.)])

    def set_plot(self, ax, **kwargs):
        ax.set_xlim(0, self._size[1])
        ax.set_ylim(0, self._size[0])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_image(self, ax, obj=None, i=None, **kwargs):
        denoisor = kwargs['denoisor']
        
        # if receive nullptr
        if i is None:
            if denoisor == 'frames':
                obj = ax.imshow(
                    np.zeros(self._size), 
                    vmin=0, vmax=255,
                    cmap=self._fr_cmap
                )   
            else:
                obj = ax.imshow(
                    np.zeros(self._size), 
                    vmin=-1, vmax=1,
                    cmap=self._ev_cmap
                )
            ax.set_title(denoisor, fontdict={'fontsize': 18, 'family': 'serif', 'weight':'normal'})
            return obj

        # if receive placeholder
        if denoisor == 'none':
            return obj
        
        # if receive frame/event data
        if denoisor == 'frames':
            frames = self._packets[i]['frames']
            if not frames.isEmpty():
                image = frames.front().image
                obj.set_data(np.flip(image, axis=0))
            return obj
        else:
            events = self._packets[i][denoisor]
            if not events.isEmpty():
                image = _visualize_events(events, self._size)
                obj.set_data(np.flip(image, axis=0))
            else:
                obj.set_data(np.zeros(self._size))
            return obj
        
    def get_ticks(self):
        return self._ticks


class MultiDenoisorsPlayer(object):
    def __init__(self, resolution: Tuple[int, int], layout: List[List[str]]):
        self.resolution = resolution
        self.layout = layout
        self.figure = mp_player.Figure()
        
    def viewPerTimeInterval(self, data, 
                            reference: str = "events",
                            interval: timedelta = timedelta(milliseconds=33)):
        # slice data
        slicer, packets = kit.MonoCameraSlicer(), list()
        slicer.doEveryTimeInterval(
            reference, interval, lambda data: packets.append(data)
        )
        slicer.accept(data)

        # run
        self._run(self.figure, Preset(self.resolution, packets))

    def viewPerNumberInterval(self, data, 
                              reference: str = "events",
                              interval: int = 15000):
        # slice data
        slicer, packets = kit.MonoCameraSlicer(), list()
        slicer.doEveryNumberOfElements(
            reference, interval, lambda data: packets.append(data)
        )
        slicer.accept(data)
        
        # run
        self._run(self.figure, Preset(self.resolution, packets))

    def _run(self, figure, preset):
        # set cols and rows
        rows, cols = len(self.layout), len(self.layout[0])

        # initialize as 2d images with (rows, cols)
        specs = [[{"type": "2d"} for i in range(cols)] for j in range(rows)]
        
        # set figure configs
        figure.set_ticks(preset.get_ticks())
        figure.set_subplot(rows=rows, cols=cols, specs=specs)
        for i, denoisors in enumerate(self.layout):
            for j, denoisor in enumerate(denoisors):
                figure.append_trace(row=i+1, col=j+1, plot_func=preset.plot_image, set_func=preset.set_plot, denoisor=denoisor)
        
        # show
        figure.show()
