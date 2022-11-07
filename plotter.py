
from typing import List
import matplotlib
import os


class Plotter():

    def __init__(self, figures_path: str):

        self.figures_path = figures_path
        
       
    def save_plots(iteration: int,
                   p_train_losses: List[float],
                   p_val_losses: List[float],
                   v_train_losses: List[float],
                   v_val_losses: List[float],
                   v_train_acc: List[float],
                   v_val_acc: List[float]):

        x = [i in range(len(p_train_losses))]

        p_loss_fig = matplotlib.figure.Figure()
        p_loss_fig_axis = p_loss_fig.add_subplot()
        p_loss_fig_axis.plot(x, p_train_losses)
        p_loss_fig_axis.plot(x, p_val_losses)
        p_loss_fig.savefig(os.path.join(fig_save_path, f'{iteration}_P_loss.png'))

        v_loss_fig = matplotlib.figure.Figure()
        v_loss_fig_axis = v_loss_fig.add_subplot()
        v_loss_fig_axis.plot(x, v_train_losses)
        v_loss_fig_axis.plot(x, v_val_losses)
        v_loss_fig.savefig(os.path.join(fig_save_path, f'{iteration}_V_loss.png'))

        v_acc_fig = matplotlib.figure.Figure()
        v_acc_fig_axis = v_acc_fig.add_subplot()
        v_acc_fig_axis.plot(x, v_train_acc)
        v_acc_fig_axis.plot(x, v_val_acc)
        v_acc_fig.savefig(os.path.join(fig_save_path, f'{iteration}_V_acc.png'))
