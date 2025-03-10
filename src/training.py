import torch; import torch.nn.functional as F; import torch.distributions as D
from torch.optim import Adam, SGD
import tqdm; import warnings; import functools; import numpy as np
from src.samplers import marginal_prob_std, diffusion_coeff
import json
warnings.filterwarnings("ignore")

sigma =  25.0; device = 'cuda'
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device = device)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device = device)

# modified from first answer in https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience, max_fraction_over, decay_points):
        self.patience = patience
        self.max_fraction_over = max_fraction_over
        self.decay_points = decay_points
        self.counter = 0
        self.min_validation_loss = float('inf')
    def early_stop(self, validation_loss, epoch_num):
        '''
        Three outcomes: (isStopped, isBetter) = (True, False), (False, False), (False, True)
        '''
        isBetter = False
        if epoch_num in self.decay_points: # tolerance could decrease as epochs increase
            self.max_fraction_over = self.max_fraction_over * 0.5
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0; isBetter = True
        elif validation_loss > self.min_validation_loss * (1 + self.max_fraction_over):
            self.counter += 1
            if self.counter >= self.patience:
                return True, isBetter
        return False, isBetter # not stopped, but might not be better

# modified from: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']