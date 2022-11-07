import torch

from alpha_zero_launcher import run_train_it
from options import AlphaZeroOptions

options = AlphaZeroOptions()
opts = options.parse()

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU!", torch.cuda.get_device_name(None))
    else:
        device = torch.device("cpu")
        print("Using CPU :(")

    run_train_it(data_path="data/structures/",
                 figures_path="data/figures/",
                 models_path="data/models/",
                 device=device,
                 args=opts)
