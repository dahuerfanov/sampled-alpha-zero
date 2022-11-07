
import numpy as np
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential, Conv2d, Module, Softmax, BatchNorm2d, \
                     Dropout, Flatten, Tanh, MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import torch

from nnet.cat_cross_entropy import CategoricalCrossEntropy


class NNet(Module):

    def __init__(self, name: str, device: str, args: Dict):
        super(NNet, self).__init__()

        self.name = name
        self.device = device
        self.args = args
        self.cnn_layers = Sequential(
            Conv2d(in_channels=2, out_channels=args.num_channels_cnn, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=args.num_channels_cnn, momentum=0.1),
            ReLU(inplace=True),
            Conv2d(in_channels=args.num_channels_cnn, out_channels=args.num_channels_cnn, kernel_size=3, stride=1,
                   padding=1),
            BatchNorm2d(num_features=args.num_channels_cnn, momentum=0.1),
            ReLU(inplace=True),
            Conv2d(in_channels=args.num_channels_cnn, out_channels=args.num_channels_cnn, kernel_size=3, stride=1,
                   padding=1),
            BatchNorm2d(num_features=args.num_channels_cnn, momentum=0.1),
            ReLU(inplace=True)
        )

        self.v_layers = Sequential(
            Conv2d(in_channels=args.num_channels_cnn, out_channels=4, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(num_features=4, momentum=0.1),
            ReLU(inplace=True),
            Flatten(),
            Linear(in_features=4 * args.rows * args.cols, out_features=32),
            Dropout(p=0.45, inplace=True),
            ReLU(inplace=True),
            Linear(in_features=32, out_features=1),
            Tanh()
        )

        self.p_layers = Sequential(
            Conv2d(in_channels=args.num_channels_cnn, out_channels=8, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(num_features=8, momentum=0.1),
            ReLU(inplace=True),
            Flatten(),
            Linear(in_features=8 * args.rows * args.cols, out_features=args.cols),
            Softmax(dim=1)
        )

        self.criterion_p = CategoricalCrossEntropy()
        self.criterion_v = MSELoss()

        self = self.to(device)
        self.criterion_p = self.criterion_p.to(device)
        self.criterion_v = self.criterion_v.to(device)

        self.optimizer = SGD(self.parameters(), lr=args.lr_sgd, momentum=args.momentum_sgd,
                             nesterov=True, weight_decay=args.wd_sgd)


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x0 = self.cnn_layers(x)
        x1 = self.v_layers(x0)
        x2 = self.p_layers(x0)
        return x1, x2


    # runs model on one sample
    def predict(self, s: Tensor) -> Tuple[float, np.array]:
        self.eval()
        with torch.no_grad():
            # create a batch of size 1
            x = torch.stack([s])
            # converting the data into GPU format (if available)
            x = x.to(self.device)
            x0 = self.cnn_layers(x)
            x1 = self.v_layers(x0)
            x2 = self.p_layers(x0)
            return x1[0].item(), x2[0].cpu().numpy()


    def train_batch(self, x_train: Tensor, v_train: Tensor, p_train: Tensor) -> Tuple[float, float, float]:

        self.train()
        # clearing the Gradients of the model parameters
        self.optimizer.zero_grad()

        # ========forward pass=====================================
        v_output_train, p_output_train = self(x_train)

        v_output_train = torch.flatten(v_output_train)
        v_loss_train = self.criterion_v(v_output_train, v_train)
        p_loss_train = self.criterion_p(p_output_train, p_train)
        loss_train = v_loss_train + p_loss_train

        # computing the updated weights of all the model parameters
        loss_train.backward()
        self.optimizer.step()

        return v_loss_train.item(), p_loss_train.item(), \
               (torch.round(v_output_train) == torch.round(v_train)).float().sum().item()


    def validate(self, x: Tensor, v: Tensor, p: Tensor) -> Tuple[float, float, float]:

        self.eval()
        # ========forward pass=====================================
        with torch.no_grad():
            v_output, p_output = self(x)
            v_output = torch.flatten(v_output)
            v_loss = self.criterion_v(v_output, v)
            p_loss = self.criterion_p(p_output, p)

            return v_loss.item(), p_loss.item(), \
                   (torch.round(v_output) == torch.round(v)).float().sum().item()


    # Tensor are gonna be moved to self.device
    def run_training(self, X: List[Tensor], Y_v: List[Tensor], Y_p: List[Tensor], test_size_ratio=0.2) -> Dict:

        assert len(X) == len(Y_v) == len(Y_p)
        sim_data = []
        for i in range(len(X)):
            sim_data.append([X[i], Y_v[i], Y_p[i]])
        trainset, testset = train_test_split(sim_data, test_size=test_size_ratio)

        trainloader = DataLoader(dataset=trainset, batch_size=self.args.batch, shuffle=True)
        testloader = DataLoader(dataset=testset, batch_size=self.args.batch, shuffle=True)

        v_train_losses = []
        p_train_losses = []
        v_val_losses = []
        p_val_losses = []

        v_train_acc = []
        v_val_acc = []

        # training the model
        for _ in range(self.args.epochs):
            loss_v_tr, loss_p_train, loss_v_val, loss_p_val = 0, 0, 0, 0
            acc_v_tr, acc_v_val = 0, 0
            num_train_batches = 0
            total_train_samples = 0
            for batch_idx, (S, V, P) in enumerate(trainloader):
                S = S.to(self.device)
                V = V.to(self.device)
                P = P.to(self.device)
                num_train_batches += 1
                total_train_samples += V.size(0)
                loss_v_tr_, loss_p_train_, acc_v_tr_ = self.train_batch(S, V, P)
                loss_v_tr += loss_v_tr_
                loss_p_train += loss_p_train_
                acc_v_tr += acc_v_tr_

            num_val_batches = 0
            total_val_samples = 0
            for batch_idx, (S, V, P) in enumerate(testloader):
                S = S.to(self.device)
                V = V.to(self.device)
                P = P.to(self.device)
                num_val_batches += 1
                total_val_samples += V.size(0)
                loss_v_val_, loss_p_val_, acc_v_val_ = self.validate(S, V, P)
                loss_v_val += loss_v_val_
                loss_p_val += loss_p_val_
                acc_v_val += acc_v_val_

            v_train_losses.append(loss_v_tr / num_train_batches)
            p_train_losses.append(loss_p_train / num_train_batches)
            v_val_losses.append(loss_v_val / num_val_batches)
            p_val_losses.append(loss_p_val / num_val_batches)

            v_train_acc.append(acc_v_tr / total_train_samples)
            v_val_acc.append(acc_v_val / total_val_samples)

        stats = {'v_train_losses': v_train_losses,
                 'p_train_losses': p_train_losses,
                 'v_val_losses': v_val_losses,
                 'p_val_losses': p_val_losses,
                 'v_train_acc': v_train_acc,
                 'v_val_acc': v_val_acc
                } 
        return stats 
