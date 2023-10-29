import time
import copy
import argparse
import numpy as np
import os.path as osp
from typing import Tuple, List
from datetime import timedelta

# Import dv related package
import dv_processing as dv
import dv_toolkit as kit

# Import project config
from configs import Denoisor

# Import player component
from python.src.utils.player import MultiDenoisorsPlayer


if __name__ == '__main__':
    # Arguments settings
    parser = argparse.ArgumentParser(description='A simple demo to verify the installation integrity.')
    parser.add_argument('-f', '--file', type=str, default='./data/demo/samples/demo-02.aedat4')
    parser.add_argument('--denoisor_list', type=list, default=[['raw', 'knoise', 'dwf', 'ynoise'], ['frames', 'ts', 'evflow', 'red']])
    args = parser.parse_args()    

    # Load file to MonoCameraReader
    reader = kit.io.MonoCameraReader(args.file)
    
    # Get Offline data
    data = reader.loadData()

    # Print basic info
    print(f"\nstarting demo, noise sequence length is {len(data['events'])}")

    # Print table title
    print(f"{'='*35}\n{'Model':15s}{'Length':10s}{'Runtime':10s}\n{'-'*35}")
    
    # Set denoisors 2d-array layout
    denoisors = args.denoisors

    # Running inference with each denoisor
    for denoisor in [elem for vec in denoisors for elem in vec]:
        # Initialize filter sequence
        filter_seq = kit.MonoCameraData()
        
        # Save noise sequence directly
        if denoisor == 'raw' or denoisor == 'none':
            data[denoisor] = data['events']
            continue
        elif denoisor == 'frames':
            continue
        
        # Time to start inference
        st_time = time.time()
        print(f"{denoisor:10s}", end=" ")

        # Initialize denoisor
        model = Denoisor(denoisor, reader.getResolution())
        # If you want to set other input parameters,
        # please check "./configs/denoisors.py"

        # Receive
        model.accept(data["events"])

        # Store denoising result
        data[denoisor] = model.generateEvents()

        # Time to end inference
        print(f"{len(data[denoisor]):10d}{time.time() - st_time:10.3f}s")
    
    # Print underline
    print(f"{'='*35}")

    # Starting processing for visualization
    print(f"Completed! Running visualization (it may takes a few minutes).")

    # running animation
    player = MultiDenoisorsPlayer(reader.getResolution(), denoisors)

    # View every 33 millisecond of events
    player.viewPerTimeInterval(data, "events", timedelta(milliseconds=33))
