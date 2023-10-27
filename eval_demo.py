import time
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from evtool.dvs import DvsFile
from evtool.utils._player.mpl_player import Animator
from configs import Denoisor


if __name__ == '__main__':
    # load file
    noise_seq = DvsFile.load('./data/demo/samples/demo-02.aedat4')
    print(f"starting demo, noise sequence length is {len(noise_seq['events'])}")

    # running inference
    print(f"{'='*35}\n{'Model':15s}{'Length':10s}{'Runtime':10s}\n{'-'*35}")
    models, dataDict = ['raw', 'knoise', 'dwf', 'ynoise', 'none', 'ts', 'evflow', 'red'], {}
    for model in models:
        if model == 'raw' or model == 'none':
            dataDict[model] = noise_seq
            continue

        st_time = time.time()
        print(f"{model:10s}", end=" ")
        dataDict[model] = Denoisor(model).run(noise_seq.copy())
        print(f"{len(dataDict[model]['events']):10d}{time.time() - st_time:10.3f}s")
    print(f"{'='*35}")

    print(f"Completed! Running visualization (it may takes a few minutes).")
    # slice events
    interval, length, size = '25ms', 1E10, noise_seq['size']
    from_timestamp = noise_seq['events'][0].timestamp
    for model, data in dataDict.items():
        if model == 'none': continue
        dataDict[model] = [ev for ev in data['events'].slice(interval, from_timestamp)]
        length = min(length, len(dataDict[model]))

    # initialize animation
    obj, m, n = [], 2, 4
    fig, axs = plt.subplots(m, n, figsize=(15, 6))
    for j, (model, data) in enumerate(dataDict.items()):
        x, y = np.unravel_index(j, (m, n))
        obj.append(axs[x][y].imshow(np.zeros(size), vmin=-1, vmax=1, cmap=plt.set_cmap('bwr')))
        if model == 'none':
            axs[x][y].set_axis_off()
        else:
            axs[x][y].set_xticks([])
            axs[x][y].set_yticks([])
            axs[x][y].xaxis.set_tick_params(labelbottom=False)
            axs[x][y].yaxis.set_tick_params(labelleft=False)

    def update(i):
        for j, (denoisor, data) in enumerate(dataDict.items()):
            x, y = np.unravel_index(j, (m, n))
            if denoisor == 'none': continue
            timestamp, event = data[i]
            obj[j].set_data(event.project(size))
            axs[x][y].set_title(denoisor, fontdict={'fontsize': 18, 'family': 'serif', 'weight':'normal'})

    # running animation
    Player = Animator(fig, update, ticks=[i for i in range(length)])
    Player.run("result.gif")
