import os
import os.path as osp
import numpy as np

from glob import glob
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, filepath):
        self.files = []
        # Search through all files
        for dirname in os.listdir(filepath):
            for filename in os.listdir(osp.join(filepath, dirname)):
                self.files.append((filepath, dirname, filename))
        self.files.sort()

    def __getitem__(self, index):
        fpath, fclass, fname = self.files[index]
        return fpath, fclass, fname
    
    def __len__(self):
        return len(self.files)


def Dataset(file_path):
    folders = [f for f in os.scandir(file_path) if f.is_dir()]
    return [BaseDataset(f.path) for f in folders]
    