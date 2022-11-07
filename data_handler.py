
from pathlib import Path
from typing import List, Tuple
from torch import Tensor
import glob
import pickle
import os
import random

from nnet.model import NNet

def save_obj(obj, obj_path):
    Path(os.path.dirname(obj_path)).mkdir(parents=True, exist_ok=True)
    with open(obj_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(obj_path):
    with open(obj_path, 'rb') as f:
        return pickle.load(f)


def save_new_samples(data_path: str,
                     samples_board: List[Tensor],
                     samples_distr: List[Tensor],
                     samples_val: List[Tensor]):

    assert len(samples_board) == len(samples_distr) == len(samples_val)
    if len(os.listdir(data_path)) > 0:
        dirs_list = [int(subdir) for subdir in os.listdir(data_path)]
        dirs_list.sort()
        max_key = dirs_list[-1]
    else:
        max_key = -1

    for board, distr, val in zip(samples_board, samples_distr, samples_val):
        max_key += 1
        save_obj(board, os.path.join(data_path, str(max_key), "board.pkl"))
        save_obj(distr, os.path.join(data_path, str(max_key), "distr.pkl"))
        save_obj(val, os.path.join(data_path, str(max_key), "val.pkl"))


class DataProvider():

    def __init__(self, data_path: str, max_num_samples_mem: int):
    
        self.data = {}
        self.max_key = 0
        self.max_num_samples_mem = max_num_samples_mem
        self.data_path = data_path


    def _load_new_samples(self) -> None:
        dirs_list = [int(subdir) for subdir in os.listdir(self.data_path)]
        dirs_list.sort(reverse=True)
        sample_key = 0
        min_key = -1
        new_max_key = -1
        if len(self.data) > 0:
            min_key = min(self.data.keys())
        for sample_key in dirs_list:
            if new_max_key < 0:
                new_max_key = sample_key
            if sample_key <= self.max_key:
                break
            self.data[sample_key] = {}
            self.data[sample_key]['board'] = load_obj(os.path.join(self.data_path, str(sample_key), "board.pkl"))
            self.data[sample_key]['distr'] = load_obj(os.path.join(self.data_path, str(sample_key), "distr.pkl"))
            self.data[sample_key]['val'] = load_obj(os.path.join(self.data_path, str(sample_key), "val.pkl"))
            if len(self.data) > self.max_num_samples_mem:
                if min_key < 0:
                    min_key = min(self.data.keys())
                self.data.pop(min_key, None)
                min_key += 1

        self.max_key = max(self.max_key, new_max_key)


    def select_samples(self, sample_size: int) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:

        self._load_new_samples()
        sample_boards, sample_distr, sample_val = [], [], []
        if len(self.data) <= sample_size:
            for s_id in self.data:
                sample_boards.append(self.data[s_id]['board'])
                sample_distr.append(self.data[s_id]['distr'])
                sample_val.append(self.data[s_id]['val'])
        else:
            for i in random.sample(self.data.keys(), sample_size):
                sample_boards.append(self.data[i]['board'])
                sample_distr.append(self.data[i]['distr'])
                sample_val.append(self.data[i]['val'])

        return sample_boards, sample_val, sample_distr
