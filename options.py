#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import argparse
import os

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)


class AlphaZeroOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Safe Area Predictor options")

        # paths
        self.parser.add_argument("--rows",
                                 type=int,
                                 help="number of board rows",
                                 default=6)

        self.parser.add_argument("--cols",
                                 type=int,
                                 help="number of board columns",
                                 default=7)

        self.parser.add_argument("--c_puct",
                                 type=float,
                                 help="Exploration coefficient in mcts",
                                 default=1)

        self.parser.add_argument("--n_threshold_exp",
                                 type=int,
                                 help="Num. of episodes at the beginning of self-play sim. for which tau = 1,afterwards tau=0",
                                 default=14)

        self.parser.add_argument("--epochs",
                                 type=int,
                                 help="Epochs for training cnn in each iteration",
                                 default=100)

        self.parser.add_argument("--momentum_sgd",
                                 type=float,
                                 help="momentum in sgd algorithm",
                                 default=0.95)

        self.parser.add_argument("--lr_sgd",
                                 type=float,
                                 help="learning rate of sgd algorithm",
                                 default=5e-3)

        self.parser.add_argument("--wd_sgd",
                                 type=float,
                                 help="weight decay of sgd algorithm",
                                 default=4e-3)

        self.parser.add_argument("--num_iters",
                                 type=int,
                                 help="maximum number of iterations of AlphaZero algorithm",
                                 default=1000)

        self.parser.add_argument("--num_eps",
                                 type=int,
                                 help="number of simulations of self-play",
                                 default=500)

        self.parser.add_argument("--num_eps_pit",
                                 type=int,
                                 help="number of simulations to pit the new and the best cnn",
                                 default=21)

        self.parser.add_argument("--batch",
                                 type=int,
                                 help="batch size",
                                 default=512)

        self.parser.add_argument("--num_mcts_sims",
                                 type=int,
                                 help="number of mcts simulations",
                                 default=35)

        self.parser.add_argument("--threshold",
                                 type=float,
                                 help="pit rate thereshold to decide the best cnn",
                                 default=0.55)

        self.parser.add_argument("--alpha_n",
                                 type=float,
                                 help="alpha constant of Dirichlet Noise in mcts",
                                 default=1)

        self.parser.add_argument("--sleep_secs_before_train",
                                 type=int,
                                 help="secs of data generation by self-play before start training",
                                 default=0)

        self.parser.add_argument("--eps_n",
                                 type=float,
                                 help="eps constant of Dirichlet Noise in mcts",
                                 default=0.25)

        self.parser.add_argument("--max_num_samples_mem",
                                 type=int,
                                 help="max. number of episodes in memory to train new cnn's",
                                 default=500000)

        self.parser.add_argument("--sample_size",
                                 type=int,
                                 help="board sample size to choose from old episodes in memory",
                                 default=10000000)

        self.parser.add_argument("--num_channels_cnn",
                                 type=int,
                                 help="nr. of channels in cnn architecture",
                                 default=256)

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="saved model to start algo from. None from scratch",
                                 default=None)


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
