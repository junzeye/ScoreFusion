JUPYTER_MODE = False
import torch; import torch.nn as nn
from scipy import integrate
import numpy as np; import os; import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
import functools; import argparse
from src.datasets import EMNISTDataset, prepare_data_digits, split_dataset
from src.samplers import marginal_prob_std, diffusion_coeff, \
    Euler_Maruyama_sampler_HD, ode_sampler_HD, Net, classify_samples
from src.model_EMNIST import ScoreNet, FusionNet
parser = argparse.ArgumentParser()
parser.add_argument("-seed", required=True, type=int, help="The random seed to use")
args = parser.parse_args(); seed = args.seed # set random seed value
print('Random seed is:', seed, '\n'); seed_directory = f'ckpt/s{seed}'
torch.manual_seed(seed); torch.cuda.manual_seed(seed) # seed the PyTorch RNG
torch.set_printoptions(precision=3, sci_mode = False, linewidth=150)
np.set_printoptions(precision=3, suppress=True, linewidth=80, threshold=1000)
import time

NUM_REPEATS = 2
sigma =  25.0; device = 'cuda'
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
# auxiliary dataset information
digits = [7, 9]
target_info = {
    'weights': [0.6, 0.4], 'train_size': -1, 'train_batch': -1, 
    'val_size': -1,'val_batch': -1,
    'test_size': 5000,'test_batch': 500 # PARAMETERS OF THIS FILE
}
test_batch_size = target_info['test_batch']
print(f'Test batch size is {test_batch_size}')
z_images, z_labels = extract_test_samples('byclass') # test
emnistdata_test = EMNISTDataset(z_images, z_labels, digits)
[test_tar_loader], [test_tar_data] = prepare_data_digits(emnistdata_test, digits, [target_info], 'test')

# Load the auxiliary models
aux_scores = nn.ModuleList() # store the score models
for d in range(4): # the auxiliary digits
    ckpt = torch.load(f'ckpt/HD/aux_hd_{d}.pth', map_location=device)
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    score_model.load_state_dict(ckpt)
    for param in score_model.parameters():
        param.requires_grad = False # freeze the gradients
    aux_scores.append(score_model)


def prior_likelihood(z, sigma):
  """The likelihood of a Gaussian distribution with mean zero and
      standard deviation sigma."""
  shape = z.shape
  N = np.prod(shape[1:])
  return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)


def ode_likelihood(x,
                   score_model,
                   marginal_prob_std,
                   diffusion_coeff,
                   batch_size=64,
                   device='cuda',
                   eps=1e-5):
  """Compute the likelihood with probability flow ODE.

  Args:
    x: Input data.
    score_model: A PyTorch model representing the score-based model.
    marginal_prob_std: A function that gives the standard deviation of the
      perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the
      forward SDE.
    batch_size: The batch size. Equals to the leading dimension of `x`.
    device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
    eps: A `float` number. The smallest time step for numerical stability.

  Returns:
    z: The latent code for `x`.
    bpd: The log-likelihoods in bits/dim.
  """

  # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
  epsilon = torch.randn_like(x)

  def divergence_eval(sample, time_steps, epsilon):
    """Compute the divergence of the score-based model with Skilling-Hutchinson."""
    with torch.enable_grad():
      sample.requires_grad_(True)
      score_e = torch.sum(score_model(sample, time_steps) * epsilon)
      grad_score_e = torch.autograd.grad(score_e, sample)[0]
    return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))

  shape = x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper for evaluating the score-based model for the black-box ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    with torch.no_grad():
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def divergence_eval_wrapper(sample, time_steps):
    """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
    with torch.no_grad():
      # Obtain x(t) by solving the probability flow ODE.
      sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
      time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
      # Compute likelihood.
      div = divergence_eval(sample, time_steps, epsilon)
      return div.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((shape[0],)) * t
    sample = x[:-shape[0]]
    logp = x[-shape[0]:]
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
    logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
    return np.concatenate([sample_grad, logp_grad], axis=0)

  init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
  # Black-box ODE solver
  res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')
  zp = torch.tensor(res.y[:, -1], device=device)
  z = zp[:-shape[0]].reshape(shape)
  delta_logp = zp[-shape[0]:].reshape(shape[0])
  sigma_max = marginal_prob_std(1.)
  prior_logp = prior_likelihood(z, sigma_max)
  bpd = -(prior_logp + delta_logp) / np.log(2)
  N = np.prod(shape[1:])
  bpd = bpd / N + 8.
  return z, bpd


# list of batch-averaged nll values - should be distributed according to the CLT
for i, aux_score in enumerate(aux_scores):
    # For each auxiliary model, compute its NLL on the target test set
    print(f'-----Auxiliary model {i} Test NLL-----')
    nll_estimators = []
    for _ in range(NUM_REPEATS):
      for x in test_tar_loader:
          x = x[0].to(device)
          # uniform dequantization
          x = ((((x * 0.3081 + 0.1307) * 255. + torch.rand_like(x)) / 256) - 0.1307) / 0.3081
          _, fusion_bpd = ode_likelihood(x, aux_score, marginal_prob_std_fn,
                                  diffusion_coeff_fn,
                                  x.shape[0], device=device, eps=1e-5)        
          nll_estimate = fusion_bpd.sum() / fusion_bpd.shape[0]
          nll_estimators.append(nll_estimate)
          x.to('cpu') # release GPU memory
    nll_estimators = torch.tensor(nll_estimators, device = 'cpu')
    print(f'mean: {nll_estimators.mean()}, std: {nll_estimators.std()}\n')