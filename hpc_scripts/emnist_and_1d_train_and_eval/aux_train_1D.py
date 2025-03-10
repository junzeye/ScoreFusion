JUPYTER_MODE = False
import sys; import os; import argparse
import tqdm; import warnings; warnings.filterwarnings("ignore")
import numpy as np; import scipy.stats as stats
import torch; import torch.nn as nn; import torch.nn.functional as F
import torch.distributions as D; from torch.optim import Adam, SGD
from torch.distributions import Gamma, Normal, Weibull, LogNormal, MixtureSameFamily
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
import numpy as np; import matplotlib.pyplot as plt; import functools
from src.model_1D import generic_train, find_test_loss
from src.samplers import marginal_prob_std, diffusion_coeff, Euler_Maruyama_sampler_1D, ode_sampler_1D
from src.datasets import make_mixture_data, Dataset_1D
from src.plotter import plot_loss
folder_paths = ['ckpt', 'out'] # create folders if they don't exist yet
for path in folder_paths:
    os.makedirs(path) if not os.path.exists(path) else None
sigma =  25.0; device = 'cuda'
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device = device)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device = device)
parser = argparse.ArgumentParser()
parser.add_argument("-seed", required=True, type=int, help="The random seed to use")
args = parser.parse_args(); seed = args.seed # set random seed value
print('Random seed is:', seed, '\n'); seed_directory = f'ckpt/s{seed}'
if not os.path.exists(seed_directory):
    os.makedirs(seed_directory)

N_EPOCHS_aux = 2048; PLT_OFFSET = 200
torch.manual_seed(seed); torch.cuda.manual_seed(seed)
aux_train_params_list = [
    {   'name': 'aux 0',
        'n_epochs': N_EPOCHS_aux, 'lr': 1e-4, 
        'patience': 50, 'max_fraction': 0.5, 'decay_points': [-1], 
        'ckpt_path': seed_directory + '/aux_1d_0.pth'
    },
    {   'name': 'aux 1',
        'n_epochs': N_EPOCHS_aux, 'lr': 1e-4, 
        'patience': 50, 'max_fraction': 0.5, 'decay_points': [-1], 
        'ckpt_path': seed_directory + '/aux_1d_1.pth'
    },
    {   'name': 'aux 2',
        'n_epochs': N_EPOCHS_aux, 'lr': 1e-4, 
        'patience': 50, 'max_fraction': 0.5, 'decay_points': [-1], 
        'ckpt_path': seed_directory + '/aux_1d_2.pth'
    }
]

# Target distribution
mix_params = {
    'means': torch.tensor([-7., 6.5]),
    'stds': torch.tensor([1., 1.]),
    'weights': torch.tensor([0.45, 0.55]),
    'train_size': 64, 'train_batch': 64,
    'val_size': 512, 'val_batch': 256,
    'test_size': 8096, 'test_batch': 1024
}

# Auxiliary distribution - three bimodal mixture of gaussians
auxs_info = [
    {
        'means': torch.tensor([7, -9.]),
        'stds': torch.tensor([1., 1.]),
        'weights': torch.tensor([0.4, 0.6]),
        'train_size': 8096, 'train_batch': 2048,
        'val_size': 2048, 'val_batch': 1024,
        'test_size': 8096, 'test_batch': 1024
    },
    {
        'means': torch.tensor([-3., 10.]),
        'stds': torch.tensor([1., 1.]),
        'weights': torch.tensor([0.5, 0.5]),
        'train_size': 8096, 'train_batch': 2048,
        'val_size': 2048, 'val_batch': 1024,
        'test_size': 8096, 'test_batch': 1024
    },
    {
        'means': torch.tensor([-5., 3.]),
        'stds': torch.tensor([1., 1.]),
        'weights': torch.tensor([0.4, 0.6]),
        'train_size': 8096, 'train_batch': 2048,
        'val_size': 2048, 'val_batch': 1024,
        'test_size': 8096, 'test_batch': 1024
    }
]

#@title Make data for training auxiliary models
VERBOSE = True # Print dataset specs
RUN_MODES = ['train', 'val', 'test']

get_mixture_data = lambda mode: make_mixture_data(Normal, mix_params, mode)
make_loader = lambda dataaset, batchSize: DataLoader(TensorDataset(dataaset), batchSize,
                                            shuffle = True, num_workers = 4)
# Load vanilla training data
(train_tar_data, train_tar_mean, train_tar_std), (val_tar_data, val_tar_mean, val_tar_std), \
    (test_tar_data, test_tar_mean, test_tar_std) = map(get_mixture_data, RUN_MODES)
train_tar_loader, val_tar_loader, test_tar_loader = \
    map(make_loader, [train_tar_data, val_tar_data, test_tar_data],
        [mix_params['train_batch'],mix_params['val_batch'],mix_params['test_batch']])
if VERBOSE:
    print('---target datasets specs---')
    print('train data info: ', (train_tar_loader.dataset.tensors[0].shape, train_tar_mean, train_tar_std))
    print('val data info: ', (val_tar_loader.dataset.tensors[0].shape, val_tar_mean, val_tar_std))
    print('test data info:', (test_tar_loader.dataset.tensors[0].shape, test_tar_mean, test_tar_std))
    print('train val test batch sizes:',
        train_tar_loader.batch_size, val_tar_loader.batch_size, test_tar_loader.batch_size)

# Load auxiliary training data
aux_datas, aux_means, aux_stds = [{name: [] for name in RUN_MODES} for i in range(3)]
train_aux_loaders, val_aux_loaders, test_aux_loaders = [],[],[]
print('\n','---aux datasets specs---')
for i, info in enumerate(auxs_info):
    print('\n',f'---auxiliary dataset set {i}---')
    get_aux_i_data = lambda runType : make_mixture_data(Normal, info, runType)
    (train_aux_data, train_aux_mean, train_aux_std), (val_aux_data, val_aux_mean, val_aux_std), \
        (test_aux_data, test_aux_mean, test_aux_std) = map(get_aux_i_data, RUN_MODES)
    store_aux_datas = lambda runType, dataa: aux_datas[runType].append(dataa)
    store_aux_means = lambda runType, mean: aux_means[runType].append(mean)
    store_aux_stds = lambda runType, std: aux_stds[runType].append(std)
    _,_,_ = map(store_aux_datas, RUN_MODES, [train_aux_data, val_aux_data, test_aux_data])
    _,_,_ = map(store_aux_means, RUN_MODES, [train_aux_mean, val_aux_mean, test_aux_mean])
    _,_,_ = map(store_aux_stds, RUN_MODES, [train_aux_std, val_aux_std, test_aux_std])
    train_aux_loaders.append(make_loader(aux_datas['train'][i], info['train_batch']))
    val_aux_loaders.append(make_loader(aux_datas['val'][i], info['val_batch']))
    test_aux_loaders.append(make_loader(aux_datas['test'][i], info['test_batch']))
    if VERBOSE:
        print(f'train, ', train_aux_loaders[i].dataset.tensors[0].shape, aux_means['train'][i], aux_stds['train'][i], train_aux_loaders[i].batch_size)
        print(f'val, ', val_aux_loaders[i].dataset.tensors[0].shape, aux_means['val'][i], aux_stds['val'][i], val_aux_loaders[i].batch_size)
        print(f'test, ', test_aux_loaders[i].dataset.tensors[0].shape, aux_means['test'][i], aux_stds['test'][i], test_aux_loaders[i].batch_size)

## train auxiliary modules
aux_train_losses, aux_val_losses, aux_ema_losses, aux_last_saved_epochs = [[] for i in range(4)]
aux_score_models = nn.ModuleList()
# train score models for each component distribution (embarrasingly parallel)
for i, v_train_params in enumerate(aux_train_params_list):
    model_name = f'aux {i} seed {seed}'
    aux_score, train_losses, val_losses, ema_losses, last_saved_epoch = \
        generic_train(JUPYTER_MODE, train_aux_loaders[i], val_aux_loaders[i], v_train_params)
    print() # make space between prints
    find_test_loss(aux_score, test_aux_loaders[i], text = model_name)
    plot_loss(train_losses, val_losses, ema_losses, last_saved_epoch, 
              offset = PLT_OFFSET, text = model_name,
              save_path = seed_directory + f'/aux_{i}_loss.png')
    aux_score_models.add_module(f'{i}', aux_score)
    _,_,_,_ = map(lambda llist, item: llist.append(item), # append training information
            [aux_train_losses, aux_val_losses, aux_ema_losses, aux_last_saved_epochs],
            [train_losses, val_losses, ema_losses, last_saved_epoch])