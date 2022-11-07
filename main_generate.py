import torch

from alpha_zero_launcher import self_play
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

    self_play(data_path="data/structures/",
              best_model_path="data/best",
              device=device,
              args=opts)
