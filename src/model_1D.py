import torch
import torch.nn as nn; from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import Gamma, Normal, Weibull, LogNormal, MixtureSameFamily
import numpy as np; import tqdm
import functools; from src.samplers import marginal_prob_std, diffusion_coeff
from src.training import ExponentialMovingAverage, EarlyStopper

sigma =  25.0; device = 'cuda'
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device = device)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device = device)

# Acknowledgement: adapted from code published by Yang Song et al. - https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing 
# changed stride and kernel size to 1 to fit the 1D case
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""
    # Original architecture is optimized for 2D image data
    # changed strides and kernel sizes to 1 to see if it makes a difference
    # essentially there is no cross pixel learning (since we only have one pixel)
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
        marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 1, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 1, stride=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 1, stride=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 1, stride=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 1, stride=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 1, stride=1, bias=False, output_padding=0)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 1, stride=1, bias=False, output_padding=0)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 1, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # when t has dimension 1, it is broadcast to x.shape[0] which is the batch size
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class FusionNet(nn.Module):
    """A fusion model that adds a terminal fully connected layer to linearly combine
        three score functions already parametrized by three UNet-based architectures.
    """
    def __init__(self, comp_scores: nn.ModuleList, fixed_lambdas = None):
        """Initialize a mixture of scores network with pre-trained score models.

        Args:
            comp_scores: A list of pre-trained score functions with __frozen__ weights
        """
        super().__init__()
        # Gaussian random feature embedding is handled by the component score models
        self.comp_scores = comp_scores
        self.num_mixtures = len(comp_scores)
        self.manual = False
        if fixed_lambdas is None:
            self.lambdas_logits = nn.Parameter(torch.randn(self.num_mixtures))
        else:
            self.lambdas_logits = fixed_lambdas
            self.manual = True

    def forward(self, x, t):
        '''
        Input shape: (batch_size, 1, 1, 1)
        Output shape: (batch_size, 1, 1, 1)
        '''
        hs = torch.zeros(x.shape[0], self.num_mixtures, device = x.device,
                         requires_grad = False) # (batch_size, num_mixtures)
        for i, s in enumerate(self.comp_scores):
            hs[:, i] = s(x, t).squeeze() # forward pass in each component score model
        if self.manual:
            lambdas = self.lambdas_logits # no softmax, just use the fixed lambdas.
        else:
            lambdas = F.softmax(self.lambdas_logits) # (num_mixtures,)
        return (hs @ lambdas)[..., None, None, None]


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a
        time-dependent score-based model.
        x: A mini-batch of train / test data. (B, 1, 1, 1)
        marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    # t is random because t ~ U[0,1] in the outmost loop of the score estimation objective
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    # z is the error term in the transition probability p_{0t}(x(t)|x(0))
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss


def fusion_loss_fn(model, x, marginal_prob_std, eps=1e-3, h = 1e-4):
  """The loss function for training nian's fusion method.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of train / test data. (B, 1, 1, 1)
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
    h: 1e-4, with eps correction got very good results
  """
  # t is random because t ~ U[0,1] in the outmost loop of the score estimation objective
  random_t = h * torch.rand(x.shape[0], device=x.device) * (1 - eps) + h * eps # (B,)
  # z is the error term in the transition probability p_{0t}(x(t)|x(0))
  z = torch.randn_like(x) # (B,1,1,1)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


def generic_train(jupyter_mode: bool, train_loader, val_loader, train_params,
                  countdown = 100, ema_decay = 0.999):
    '''Boilerplate train function to be used both for training vanilla diffusion and
    auxiliary diffusion models (vanilla diffusion models with a lot more data)

    Args:
        train_params: dict
            n_epochs: max number of training epochs
            lr: learning rate
            patience: to EarlyStopper object. Number of times max_fraction allowed to be crossed
            max_fraction: to EarlyStopper object. Fractional threshold for increase in validation loss
        countdown: how many more epochs to train, to probe overfitting phenomenon
    Returns:
        score, train_losses, val_losses, ema_losses, last_saved_epoch
    '''
    n_epochs, lr, patience, max_fraction, decay_points, ckpt_path = \
        [train_params[key] for key in ['n_epochs', 'lr', 'patience', 'max_fraction', 'decay_points', 'ckpt_path']]
    score = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn)).to(device)
    ema_score = ExponentialMovingAverage(score.parameters(), decay = ema_decay)
    optimizer = Adam(score.parameters(), lr=lr)
    val_iter = iter(val_loader)
    train_losses, val_losses, ema_losses = [], [], []
    early_stopper = EarlyStopper(patience=patience,
                                        max_fraction_over=max_fraction,
                                        decay_points=decay_points)
    earlyStopTriggered, last_saved_epoch = False, -1
    tqdm_epoch = tqdm.notebook.trange(n_epochs) if jupyter_mode else range(n_epochs)

    for epoch_num, epoch in enumerate(tqdm_epoch):
        if earlyStopTriggered: # start the countdown once there is sign of overfitting
            countdown -= 1
        for x in train_loader:
            x = x[0].to(device)
            loss = loss_fn(score, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_score.update(score.parameters())
        # calculate EMA val loss
        ema_score.store(score.parameters())  # Cache non-EMA weights
        ema_score.copy_to(score.parameters())  # Replace with EMA weights
        with torch.no_grad():
            try:
                x_val = next(val_iter)
            except StopIteration: # Reinitialize the iterator and fetch the next batch
                val_iter = iter(val_loader)
                x_val = next(val_iter)
            x_val = x_val[0].to(device)
            ema_val_loss = loss_fn(score, x_val, marginal_prob_std_fn)
            doStop, gotBetter = early_stopper.early_stop(ema_val_loss, epoch_num)            
            if not doStop: # save if validation loss is not too high
                torch.save(score.state_dict(), ckpt_path)
                last_saved_epoch = epoch_num
        ema_score.restore(score.parameters())
        # calculate non-ema val loss
        with torch.no_grad():
            val_loss = loss_fn(score, x_val, marginal_prob_std_fn)
        train_losses.append(loss.item()); val_losses.append(val_loss.item()); ema_losses.append(ema_val_loss.item())
        if jupyter_mode:
            tqdm_epoch.set_description(f'Train Loss: {loss.item():.4f}; Val Loss: {val_loss.item():.4f}; EMA val Loss: {ema_val_loss.item():.4f}')
        if doStop and not earlyStopTriggered: # if earlystop threshold reached, then start the countdown
            earlyStopTriggered = True
        if earlyStopTriggered and countdown == 0:
            break
    score = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn)).to(device)
    score.load_state_dict(torch.load(ckpt_path, map_location=device))

    return score, train_losses, val_losses, ema_losses, last_saved_epoch


def exp_train(jupyter_mode: bool, train_loader, val_loader, train_params,
                  countdown = 100, ema_decay = 0.999):
    '''experimential train function to be used both for training vanilla diffusion and
    auxiliary diffusion models (vanilla diffusion models with a lot more data).
    For alpha-testing before incorporated into generic_train()

    Args:
        train_params: dict
            n_epochs: max number of training epochs
            lr: learning rate
            patience: to EarlyStopper object. Number of times max_fraction allowed to be crossed
            max_fraction: to EarlyStopper object. Fractional threshold for increase in validation loss
        countdown: how many more epochs to train, to probe overfitting phenomenon
    Returns:
        score, train_losses, val_losses, ema_losses, last_saved_epoch
    '''
    n_epochs, lr, patience, max_fraction, decay_points, ckpt_path = \
        [train_params[key] for key in ['n_epochs', 'lr', 'patience', 'max_fraction', 'decay_points', 'ckpt_path']]
    score = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn)).to(device)
    ema_score = ExponentialMovingAverage(score.parameters(), decay = ema_decay)
    optimizer = Adam(score.parameters(), lr=lr)
    val_iter = iter(val_loader)
    train_losses, val_losses, ema_losses = [], [], []
    early_stopper = EarlyStopper(patience=patience,
                                        max_fraction_over=max_fraction,
                                        decay_points=decay_points)
    earlyStopTriggered, last_saved_epoch = False, -1
    tqdm_epoch = tqdm.notebook.trange(n_epochs) if jupyter_mode else range(n_epochs)

    for epoch_num, epoch in enumerate(tqdm_epoch):
        if earlyStopTriggered: # start the countdown once there is sign of overfitting
            countdown -= 1
        for x in train_loader:
            x = x[0].to(device)
            loss = loss_fn(score, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_score.update(score.parameters())
        # calculate EMA val loss
        ema_score.store(score.parameters())  # Cache non-EMA weights
        ema_score.copy_to(score.parameters())  # Replace with EMA weights
        with torch.no_grad():
            try:
                x_val = next(val_iter)
            except StopIteration: # Reinitialize the iterator and fetch the next batch
                val_iter = iter(val_loader)
                x_val = next(val_iter)
            x_val = x_val[0].to(device)
            ema_val_loss = loss_fn(score, x_val, marginal_prob_std_fn)
            doStop, gotBetter = early_stopper.early_stop(ema_val_loss, epoch_num)
            if gotBetter and not earlyStopTriggered:
                torch.save(score.state_dict(), ckpt_path)
                last_saved_epoch = epoch_num
        ema_score.restore(score.parameters())
        # calculate non-ema val loss
        with torch.no_grad():
            val_loss = loss_fn(score, x_val, marginal_prob_std_fn)
        train_losses.append(loss.item()); val_losses.append(val_loss.item()); ema_losses.append(ema_val_loss.item())
        if jupyter_mode:
            tqdm_epoch.set_description(f'Train Loss: {loss.item():.4f}; Val Loss: {val_loss.item():.4f}; EMA val Loss: {ema_val_loss.item():.4f}')
        if doStop and not earlyStopTriggered: # if earlystop threshold reached, then start the countdown
            earlyStopTriggered = True
        if earlyStopTriggered and countdown == 0:
            break
        torch.save(score.state_dict(), ckpt_path)
        last_saved_epoch = epoch_num

    score = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn)).to(device)
    score.load_state_dict(torch.load(ckpt_path, map_location=device))

    return score, train_losses, val_losses, ema_losses, last_saved_epoch


def find_test_loss(score, test_loader, text = '', num_repeats = 5):
    test_losses = []
    for i in range(num_repeats): # default: 50 random samples of 512 images from the test dataset
        for x in test_loader:
            x = x[0].to(device)
            with torch.no_grad():
                loss = loss_fn(score, x, marginal_prob_std_fn).item()
                test_losses.append(loss)
    test_losses = np.array(test_losses)
    print(f"----model {text} test loss----")
    print(f"mean: {np.mean(test_losses)}, std: {np.std(test_losses)}")
    return