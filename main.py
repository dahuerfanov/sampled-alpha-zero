import torch

from alpha_zero_launcher import policy_iteration
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

    policy_iteration(data_path="data/structures/",
                     models_path="data/models/",
                     figures_path="data/figures/",
                     device=device,
                     args=opts)
