# Define the SDE/ODE backward samplers
# Acknowledgement: adapted from code published by Yang Song et al. - score_sde_pytorch
import torch; import torch.nn as nn
from scipy import integrate
import tqdm as tqdm
import torch.nn.functional as F
import numpy as np; from scipy import integrate
from torch.distributions import Categorical

num_steps =  1000 # The number of sampling steps. Applies to Euler-Maruyama and PC samplers
signal_to_noise_ratio = 0.16 # Predictor-Corrector Sampler

def marginal_prob_std(t, sigma, device='cuda'):
  """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.
  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The standard deviation.
  """
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device='cuda'):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)


# Euler-Maruyama Sampler
def Euler_Maruyama_sampler_1D(score_model,
                            marginal_prob_std,
                            diffusion_coeff,
                            batch_size=64,
                            num_steps=num_steps,
                            device='cuda',
                            eps=1e-3, jupyter_mode=False):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

    Returns:
        Samples.
    """
    t = torch.ones(batch_size, device=device)
    # Changed dimensions here: old (1, 28, 28) -> new (1, 1, 1)
    init_x = torch.randn(batch_size, 1, 1, 1, device=device) \
        * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    time_steps = tqdm.notebook.tqdm(time_steps) if jupyter_mode else time_steps
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    # Do not include any noise in the last sampling step.
    return mean_x


## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5
def ode_sampler_1D(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                z=None,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that returns the standard deviation
        of the perturbation kernel.
        diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        atol: Tolerance of absolute errors.
        rtol: Tolerance of relative errors.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
        eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        # �����dimensions���޸���: (1, 28, 28) -> (1, 1, 1)
        init_x = torch.randn(batch_size, 1, 1, 1, device=device) \
        * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x


def Euler_Maruyama_sampler_HD(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=num_steps,
                           device='cuda',
                           eps=1e-3, jupyter_mode=False):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  time_steps = tqdm.notebook.tqdm(time_steps) if jupyter_mode else time_steps
  with torch.no_grad():
    for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
  # Do not include any noise in the last sampling step.
  return mean_x

def Euler_Maruyama_sampler_celeba(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=num_steps,
                           device='cuda',
                           eps=1e-3, jupyter_mode=False):
  """Generate samples from score-based models with the Euler-Maruyama solver.
  **Assumes that the input data is 3x64x64**
  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 3, 64, 64, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  time_steps = tqdm.notebook.tqdm(time_steps) if jupyter_mode else time_steps
  with torch.no_grad():
    for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
  # Do not include any noise in the last sampling step.
  return mean_x

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5
def ode_sampler_HD(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                eps=1e-3,jupyter_mode=False):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x


# SpinalNet MNIST Classifier
first_HL =10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, first_HL) #changed from 16 to 8
        self.fc1_1 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_2 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_3 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_4 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_5 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc2 = nn.Linear(first_HL*6, 10) # changed first_HL from second_HL

        #self.fc1 = nn.Linear(320, 50)
        #self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x1 = x[:, 0:160]

        x1 = F.relu(self.fc1(x1))
        x2= torch.cat([ x[:,160:320], x1], dim=1)
        x2 = F.relu(self.fc1_1(x2))
        x3= torch.cat([ x[:,0:160], x2], dim=1)
        x3 = F.relu(self.fc1_2(x3))
        x4= torch.cat([ x[:,160:320], x3], dim=1)
        x4 = F.relu(self.fc1_3(x4))
        x5= torch.cat([ x[:,0:160], x4], dim=1)
        x5 = F.relu(self.fc1_4(x5))
        x6= torch.cat([ x[:,160:320], x5], dim=1)
        x6 = F.relu(self.fc1_5(x6))


        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)

        x = self.fc2(x)

        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        return F.log_softmax(x)


def classify_samples(network, samples, batch_size, sample_size = 512):
    network.eval()
    with torch.no_grad():
        output = network(samples)
        distribution = Categorical(logits=output)
        samples = distribution.sample((sample_size,))
        # output = network(samples)
        # pred = output.data.max(1, keepdim=True)[1]
    print(f'Digits count: {(torch.bincount(samples.ravel()) / (batch_size * sample_size)).cpu().numpy()}')


def pc_sampler(score_model,
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64,
               num_steps=num_steps,
               snr=signal_to_noise_ratio,
               device='cuda',
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

    Returns:
        Samples.
    """
    t = torch.ones(batch_size, device=device)
    # �����dimensions���޸���: (1, 28, 28) -> (1, 1, 1)
    init_x = torch.randn(batch_size, 1, 1, 1, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        # could add tqdm.tqdm(time_steps) for a progress bar
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

        # The last step does not include any noise
        return x_mean

