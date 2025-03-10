import torch; from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.utils import make_grid
from emnist import list_datasets, extract_training_samples, extract_test_samples
import torch.distributions as D
from sklearn.model_selection import train_test_split
from torch.distributions import Gamma, Normal, Weibull, LogNormal, MixtureSameFamily
import numpy as np; import copy; import matplotlib.pyplot as plt

## Methods and classes to make 1D data ##
def make_mixture_data(comp_prob, mixture_params, runtype: str):
    '''
    train_size: total number of data points to sample from
    batch_size: number of data points in a single batch
    runtype: ['train', 'val', 'test']
    '''
    assert runtype in ['train', 'val', 'test'], "invalid type!"
    param_names_dict =  {
        'train': ['train_size', 'train_batch'], 
        'val':   ['val_size', 'val_batch'], 
        'test':  ['test_size', 'test_batch']
        }; param_names = param_names_dict[runtype]
                                                                                        
    dataset_size, batch_size = mixture_params[param_names[0]], mixture_params[param_names[1]]
    sampler = MixtureSameFamily(D.Categorical(mixture_params['weights']),
                                comp_prob(mixture_params['means'], mixture_params['stds']))
    data = sampler.sample((dataset_size,))
    mean = data.mean()
    std = data.std()
    normed_data = (data - mean) / std # normalize the data
    normed_data = normed_data[:, None, None, None]
    return normed_data, mean, std

def make_data(distribution_name, params, train_size, batch_size):
    '''
    "Deprecated"
    train_size: total number of data points to sample from
    batch_size: number of data points in a single batch
    '''
    params = [torch.tensor(x, dtype = torch.float32) for x in params]
    sampler = distribution_name(*params)
    data = sampler.sample((train_size // batch_size, batch_size))
    mean = data.mean()
    std = data.std()
    normed_data = (data - mean) / std # Standardize the data
    normed_data = normed_data[:, :, None, None, None]
    return normed_data, mean, std

# 1D dataset
class Dataset_1D(Dataset):
    def __init__(self, raw_data):
        """
        Args:
        raw_data: [dataset_size,1,1,1]
        """
        # applies pytorch transforms
        self.data = raw_data
        self.targets = torch.zeros((raw_data.shape[0]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Here, you might want to add other transformations (e.g., ToTensor, which is redundant here)
        return self.data[index], self.targets[index]

    def shuffle(self): # shuffle data after recombination
        indices = torch.randperm(len(self.targets))
        self.data = self.data[indices]
        self.targets = self.targets[indices]


# Methods and classes to process MNIST data -> high-dimensional & real-world
class EMNISTDataset(Dataset):
    def __init__(self, images, targets, only_these = None, manual_entry = None):
        """
        Args:
            images (numpy.ndarray): ndarrays of images
            targets (torch.Tensor): Tensor of targets (labels)
            only_these: specific targets IDs to keep; discard the rest
        """
        if manual_entry is None:
            # applies pytorch transforms
            self.data = torch.from_numpy(images.astype(np.float32) / 255.0)
            self.data = self.data.unsqueeze(1) # add extra channel dimension
            # rescale by the mean and std of the MNIST dataset
            self.data = (self.data - torch.tensor(0.1307)) / torch.tensor(0.3081)
            self.targets = torch.from_numpy(targets.astype(np.int64))
        else:
            self.data, self.targets = images, targets

        if only_these is not None:
            indices = torch.isin(self.targets, torch.tensor(only_these))
            self.data, self.targets = self.data[indices], self.targets[indices]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Here, you might want to add other transformations (e.g., ToTensor, which is redundant here)
        return self.data[index], self.targets[index]

    def shuffle(self): # shuffle data after recombination
        indices = torch.randperm(len(self.targets))
        self.data = self.data[indices]
        self.targets = self.targets[indices]

type_dict = {
    'train': ['train_size', 'train_batch'],
    'val': ['val_size', 'val_batch'],
    'test': ['test_size', 'test_batch']
}
def prepare_data_digits(dat_dataset, digits, comp_info, run_type):
    '''
    run_type: str {'train', 'test'}
    '''
    data_sets, data_loaders, pure_datas = [], [], []
    kwds = type_dict[run_type]

    for d in digits:
        dataset = copy.deepcopy(dat_dataset)
        indices = dataset.targets == d
        dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
        dataset.shuffle()
        pure_datas.append(dataset)
    for info in comp_info:
        weights, data_size, batch_size = info['weights'], info[kwds[0]], info[kwds[1]]
        auxs = []
        for i, _ in enumerate(pure_datas):
            dataset = copy.deepcopy(pure_datas[i])
            dataset.shuffle()
            end_idx = int(weights[i] * data_size)
            dataset.data, dataset.targets = dataset.data[:end_idx], dataset.targets[:end_idx]
            auxs.append(dataset)
        combined_data = torch.cat([aux.data for aux in auxs], dim = 0)
        combined_targets = torch.cat([aux.targets for aux in auxs], dim = 0)
        combined_Dataset = copy.deepcopy(pure_datas[0])
        combined_Dataset.data, combined_Dataset.targets = combined_data, combined_targets
        loader = DataLoader(combined_Dataset, batch_size= batch_size,
                            shuffle=True, num_workers=4)
        data_sets.append(combined_Dataset); data_loaders.append(loader)
    return data_loaders, data_sets

def split_dataset(dataset, val_size, shuffle=True):
    """
    Split the dataset into training and validation sets in a stratified manner.

    Args:
        dataset (EMNISTDataset): The dataset to be split.
        val_size (int): Absolute number of the dataset to use for validation.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        (Subset, Subset): The training and validation subsets.
    """
    targets = dataset.targets.numpy()
    # Stratified split
    train_indices, val_indices = train_test_split(range(len(targets)), test_size=val_size, shuffle=shuffle)    
    # Create new datasets by subsetting manually
    train_images = dataset.data[train_indices]
    train_targets = dataset.targets[train_indices]
    val_images = dataset.data[val_indices]
    val_targets = dataset.targets[val_indices]

    # Instantiate EMNISTDataset objects for train and validation datasets
    train_dataset = EMNISTDataset(train_images, train_targets, manual_entry = True)
    val_dataset = EMNISTDataset(val_images, val_targets, manual_entry = True)

    return train_dataset, val_dataset

# debugging...
if __name__ == '__main__':
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

    x_images, x_labels = extract_training_samples('byclass') # train
    z_images, z_labels = extract_test_samples('byclass') # test

    # clumsy but hopefully works
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


    # digits_samples_train = [next(iter(train_aux_loaders[i]))[0] for i, info in enumerate(auxs_info)] 
    # digits_samples_val = [next(iter(val_aux_loaders[i]))[0] for i, info in enumerate(auxs_info)]

    # for i in range(4):
    #     print(torch.unique(train_aux_loaders[i].dataset.targets, return_counts=True))
    #     print(torch.unique(val_aux_loaders[i].dataset.targets, return_counts=True))
    #     sample_grid = make_grid(digits_samples_train[i][:64], nrow=8)
    #     plt.figure(figsize=(6,6)); plt.axis('off')
    #     plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    #     plt.savefig(f'sample_grid_train_{i}.png')
    #     plt.close()

    #     sample_grid = make_grid(digits_samples_val[i][:64], nrow=8)
    #     plt.figure(figsize=(6,6)); plt.axis('off')
    #     plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    #     plt.savefig(f'sample_grid_val_{i}.png')
    #     plt.close()