from torch import Tensor
import torch

EPS = 1e-7


class _Loss(torch.nn.Module):

    def __init__(self) -> None:
        super(_Loss, self).__init__()


class CategoricalCrossEntropy(_Loss):

    def __init__(self) -> None:
        super(CategoricalCrossEntropy, self).__init__()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        batch_sz = x.size(dim=0)
        return -torch.sum( x * torch.log(target + EPS), 1) / bath_sz
