JUPYTER_MODE = False
import torch; import torch.nn as nn
import numpy as np; import os; import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
import functools; import argparse
from src.datasets import EMNISTDataset, prepare_data_digits, split_dataset
from src.model_EMNIST import generic_train, find_test_loss
from src.plotter import plot_loss
parser = argparse.ArgumentParser()
parser.add_argument("-seed", required=True, type=int, help="The random seed to use")
args = parser.parse_args(); seed = args.seed # set random seed value
print('Random seed is:', seed, '\n'); seed_directory = f'ckpt/s{seed}'
if not os.path.exists(seed_directory):
    os.makedirs(seed_directory)
torch.manual_seed(seed); torch.cuda.manual_seed(seed) # seed the PyTorch RNG

# auxiliary dataset information
digits = [7, 9]
TRAIN_SIZE = 25000; VAL_SIZE = 5000; TEST_SIZE = 5000 # auxiliary training sizes
auxs_info = [
        {'weights': [0.1, 0.9],'train_size': TRAIN_SIZE,'train_batch': 200,'val_size': VAL_SIZE,'val_batch': 200,
        'test_size': TEST_SIZE,'test_batch': 200},
        {'weights': [0.3, 0.7],'train_size': TRAIN_SIZE,'train_batch': 200,'val_size': VAL_SIZE,'val_batch': 200,
        'test_size': TEST_SIZE,'test_batch': 200},
        {'weights': [0.7, 0.3],'train_size': TRAIN_SIZE,'train_batch': 200,'val_size': VAL_SIZE,'val_batch': 200,
        'test_size': TEST_SIZE,'test_batch': 200},
        {'weights': [0.9, 0.1],'train_size': TRAIN_SIZE,'train_batch': 200,'val_size': VAL_SIZE,'val_batch': 200,
        'test_size': TEST_SIZE,'test_batch': 200}]

# auxiliary training params
N_EPOCHS = 600
aux_train_params_list = [
    {   'name': 'HD aux 0',
        'n_epochs': N_EPOCHS, 'lr': 1e-4, 
        'patience': 50, 'max_fraction': 0.5, 'decay_points': [-1], 
        'ckpt_path': seed_directory + '/aux_hd_0.pth'
    },
    {   'name': 'HD aux 1',
        'n_epochs': N_EPOCHS, 'lr': 1e-4, 
        'patience': 50, 'max_fraction': 0.5, 'decay_points': [-1], 
        'ckpt_path': seed_directory + '/aux_hd_1.pth'
    },
    {   'name': 'HD aux 2',
        'n_epochs': N_EPOCHS, 'lr': 1e-4, 
        'patience': 50, 'max_fraction': 0.5, 'decay_points': [-1], 
        'ckpt_path': seed_directory + '/aux_hd_2.pth'
    },
    {   'name': 'HD aux 3',
        'n_epochs': N_EPOCHS, 'lr': 1e-4, 
        'patience': 50, 'max_fraction': 0.5, 'decay_points': [-1], 
        'ckpt_path': seed_directory + '/aux_hd_3.pth'
    }]
train_indices = [0, 1, 2, 3] # aux models to train

x_images, x_labels = extract_training_samples('byclass') # train
z_images, z_labels = extract_test_samples('byclass') # test
# clumsy but works
d0_data = EMNISTDataset(x_images, x_labels, [digits[0]]); d0_data.shuffle()
d0_train, d0_val = split_dataset(d0_data, VAL_SIZE)
d1_data = EMNISTDataset(x_images, x_labels, [digits[1]]); d1_data.shuffle()
d1_train, d1_val = split_dataset(d1_data, VAL_SIZE)
emnistdata_train, emnistdata_val = EMNISTDataset(x_images, x_labels, digits), EMNISTDataset(x_images, x_labels, digits)
emnistdata_train.data, emnistdata_val.data = torch.cat([d0_train.data, d1_train.data], dim = 0), \
    torch.cat([d0_val.data, d1_val.data], dim = 0)
emnistdata_train.targets, emnistdata_val.targets = torch.cat([d0_train.targets, d1_train.targets], dim = 0), \
    torch.cat([d0_val.targets, d1_val.targets], dim = 0)
emnistdata_train.shuffle(); emnistdata_val.shuffle()
train_aux_loaders,train_aux_datas = prepare_data_digits(emnistdata_train, digits, auxs_info, 'train')
val_aux_loaders,val_aux_datas = prepare_data_digits(emnistdata_val, digits, auxs_info, 'val')
emnistdata_test = EMNISTDataset(z_images, z_labels, digits)
test_aux_loaders,test_aux_datas = prepare_data_digits(emnistdata_test, digits, auxs_info, 'test')
print(x_images.shape, x_labels.shape)
print([train_aux_loaders[i].dataset.data.shape for i in range(4)])
print([val_aux_loaders[i].dataset.data.shape for i in range(4)])
print([test_aux_loaders[i].dataset.data.shape for i in range(4)])

# Train auxiliary models
for i, (train_loader, val_aux_loader, train_params) in enumerate(
    zip(train_aux_loaders, val_aux_loaders, aux_train_params_list)):
    print(f'\n-----Component model {i}-----')
    if i not in train_indices: # only train select component models
        print('Skipped...')
        continue
    print('Training...')
    model, train_losses, val_losses, ema_losses, last_saved_epoch = \
        generic_train(JUPYTER_MODE, train_loader, val_aux_loader, train_params)
    print('Testing...')
    find_test_loss(model, val_aux_loader, text = f'HD aux {i}', num_repeats = 5)
    print('Plotting...')
    plot_loss(train_losses, val_losses, ema_losses, last_saved_epoch, 
              offset = int(last_saved_epoch/10),
              text = train_params['name'], save_path = train_params['ckpt_path'][:-4] + '_loss.png')