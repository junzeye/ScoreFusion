JUPYTER_MODE = False # False if running scripts on cluster
import os; import argparse
import torch; import torch.nn as nn; from torch.optim import Adam, SGD; from torchvision.utils import make_grid
# import torch.nn.functional as F; import torchvision; from tqdm.notebook import trange
import numpy as np; import os; import matplotlib.pyplot as plt
from emnist import list_datasets, extract_training_samples, extract_test_samples
import functools; import argparse; import copy
# from src.training import ExponentialMovingAverage, EarlyStopper
from src.datasets import EMNISTDataset, prepare_data_digits, split_dataset
from src.model_EMNIST import ScoreNet, generic_train, find_test_loss, \
    loss_fn, fusion_loss_fn
from src.samplers import marginal_prob_std, diffusion_coeff, \
    Euler_Maruyama_sampler_HD, ode_sampler_HD, Net, classify_samples
from src.plotter import plot_loss

parser = argparse.ArgumentParser()
parser.add_argument("-seed", required=True, type=int, help="The random seed to use")
parser.add_argument("-trainSize", required=True, type=int, help="Train set size")
parser.add_argument("-valSize", required=True, type=int, help="Val set size")
parser.add_argument("-lr", required=True, type=float, help="Train LR")
parser.add_argument("-nEpochs", required=True, type=int, help="Number of Epochs")
args = parser.parse_args()
seed = args.seed; TRAIN_TAR_SIZE = args.trainSize; VAL_TAR_SIZE = args.valSize
train_batch = TRAIN_TAR_SIZE if TRAIN_TAR_SIZE < 128 else 128 # if data size is too small, do whole batch gradient descent
val_batch = VAL_TAR_SIZE if VAL_TAR_SIZE < 128 else 128 # if data size is too small, do whole batch gradient descent
LR = args.lr; N_EPOCHS_TAR = args.nEpochs
print('\nRandom seed is:', seed); seed_directory = f'ckpt/s{seed}'
print('Train size:', TRAIN_TAR_SIZE, 'Val size:', VAL_TAR_SIZE, 
      'LR:', LR, 'N_EPOCHS_TAR:', N_EPOCHS_TAR)
if not os.path.exists(seed_directory):
    os.makedirs(seed_directory)
torch.manual_seed(seed); torch.cuda.manual_seed(seed) # seed the PyTorch RNG
sigma =  25.0; device = 'cuda'
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device = device)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device = device)


# target training
digits = [7, 9]
target_info = {
    'weights': [0.6, 0.4], 'train_size': TRAIN_TAR_SIZE, 'train_batch': train_batch, 
    'val_size': VAL_TAR_SIZE,'val_batch': val_batch,
    'test_size': 5000,'test_batch': 200
}

# target_data prep
x_images, x_labels = extract_training_samples('byclass') # train
z_images, z_labels = extract_test_samples('byclass') # test
d0_data = EMNISTDataset(x_images, x_labels, [digits[0]]); d0_data.shuffle()
d0_train, d0_val = split_dataset(d0_data, VAL_TAR_SIZE)
d1_data = EMNISTDataset(x_images, x_labels, [digits[1]]); d1_data.shuffle()
d1_train, d1_val = split_dataset(d1_data, VAL_TAR_SIZE)
emnistdata_train, emnistdata_val = EMNISTDataset(x_images, x_labels, digits), EMNISTDataset(x_images, x_labels, digits)
emnistdata_train.data, emnistdata_val.data = torch.cat([d0_train.data, d1_train.data], dim = 0), \
    torch.cat([d0_val.data, d1_val.data], dim = 0)
emnistdata_train.targets, emnistdata_val.targets = torch.cat([d0_train.targets, d1_train.targets], dim = 0), \
    torch.cat([d0_val.targets, d1_val.targets], dim = 0)
emnistdata_train.shuffle(); emnistdata_val.shuffle()
[train_tar_loader], [train_tar_data] = prepare_data_digits(emnistdata_train, digits, [target_info], 'train')
[val_tar_loader], [val_tar_data] = prepare_data_digits(emnistdata_val, digits, [target_info], 'val')
emnistdata_test = EMNISTDataset(z_images, z_labels, digits)
[test_tar_loader], [test_tar_data] = prepare_data_digits(emnistdata_test, digits, [target_info], 'test')

print(x_images.shape, x_labels.shape)
print(train_tar_loader.dataset.data.shape)
print(val_tar_loader.dataset.data.shape)
print(test_tar_loader.dataset.data.shape)
# Load SpinalNet classifier - for later classification
network = Net()
network = network.to(device)
ckpt = torch.load(f'ckpt/emnist_classifier.pth', map_location=device)
network.load_state_dict(ckpt)

# baseline model training
tar_train_params = {
    'name': 'baseline',
    'n_epochs': N_EPOCHS_TAR, 'lr': LR,
    'patience': 50, 'max_fraction': 0.5, 'decay_points': [-1],
    'ckpt_path': seed_directory + f'/base_hd_{TRAIN_TAR_SIZE}_{VAL_TAR_SIZE}.pth'
}
baseline, base_train_losses, base_val_losses, base_ema_losses, base_last_saved_epoch = \
    generic_train(JUPYTER_MODE, train_tar_loader, val_tar_loader, tar_train_params)
print('Testing...')
find_test_loss(baseline, test_tar_loader, text = f'HD base {TRAIN_TAR_SIZE}', num_repeats = 5)
print('Plotting...')
plot_loss(base_train_losses, base_val_losses, base_ema_losses, base_last_saved_epoch,
            offset = int(base_last_saved_epoch/5),
            text = tar_train_params['name'], save_path = seed_directory + f'/base_hd_{TRAIN_TAR_SIZE}_{VAL_TAR_SIZE}' + '_loss.png')


## Generate samples using the specified sampler.
sample_batch_size = 512
sampler = Euler_Maruyama_sampler_HD
undertrain_samples = sampler(baseline,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  device=device)
sample_grid = make_grid(undertrain_samples[:64] * 0.3081 + 0.1307, nrow=8)
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu())
plt.show()
plt.savefig(seed_directory + f'/base_hd_{TRAIN_TAR_SIZE}' + '_samples.png')
# classify samples by percent digits generated
classify_samples(network,undertrain_samples, 512)