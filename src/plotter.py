import matplotlib.pyplot as plt; import numpy as np
import torch

def plot_loss(train_losses, val_losses, ema_losses, 
                last_saved_epoch, offset = 100, text = '', save_path = None):
    plt.clf()
    time_idx = np.arange(offset, len(train_losses))
    plt.plot(time_idx, ema_losses[offset:], label = 'EMA val loss', color = 'green')
    plt.plot(time_idx, val_losses[offset:], 
             label = 'val loss', color = 'darkorange')
    plt.plot(time_idx, train_losses[offset:], 
             label = 'train loss', color = 'blue', alpha = 0.5)
    # plt.xlim(left = 400); plt.ylim(top = 200, bottom = 0)
    # plt.yscale('log')
    plt.axvline(last_saved_epoch, linestyle=':', label='last saved checkpoint', color='r')
    plt.xlabel('epochs'); plt.ylabel('score loss')
    plt.title(f'{text} score loss dynamics')
    plt.grid(True); plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    return