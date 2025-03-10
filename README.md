<div align="center">
  <h1>ScoreFusion: Fusing Score-based Generative Models via Kullback-Leibler Barycenters</h1>
  <p>Code companion for the same-named paper, available at <a href="https://arxiv.org/abs/2406.19619">arXiv</a>.</p>
</div>

<div align="center">

![Visual Interpolation by the KL Barycenter Sampler - 1](media/kl_spectrum_1.jpg)
![Visual Interpolation by the KL Barycenter Sampler - 2](media/kl_spectrum_2.jpg)

Images sampled from the KL-divergence barycenter of two auxiliary models at various interpolation weights. The same Gaussian noise was used to seed the diffusion process for all setups. From left to right, column-wise: $\lambda \in [0, 0.2, 0.4, 0.6, 0.8, 1]$.

</div>

## Create a Python Virtual Environment
Create a Python virtual environment and install the required packages:

```bash
# Create a virtual environment
python -m venv your_venv_name

# Activate the virtual environment
source your_venv_name/bin/activate  # On Linux/Mac
# or
your_venv_name\Scripts\activate     # On Windows

pip install -r requirements.txt
```

## Usage
### Configure the Working Directory
**IMPORTANT**: whether you are running a Python script or a Jupyter notebook, make sure your session's current working directory is the root of the project directory (the directory containing this README file) before running any script or notebook.

### Reproduce the Results
Most experiments can be reproduced by running Jupyter notebooks in the `notebooks/` folder; their names should be self-explanatory of the experiment steps they reproduce, but feel free to contact us in case of doubt. For generating large number of images, especially for the SDXL experiments, it is recommended to run the Python scripts in the `hpc_scripts/` folder and submit the jobs to a GPU cluster. We have included template `.slurm` scripts for this purpose in the `hpc_scripts/` folder, if your cluster uses SLURM as the job scheduler.

The folder `out/` will store outputs like generated image samples and SLURM job logs. `cache/` stores some intermediate results (CLIP distance values, PyTorch model checkpoints, etc.) we used to generate the paper figures. `src/` contains modules that are shared across different Jupyter notebooks and Python scripts.

### Citation
If you find this work useful, please cite it as follows:

```bibtex
@article{scorefusion,
	title={{S}core{F}usion: {F}using {S}core-based Generative Models via {K}ullback-{L}eibler Barycenters},
	author={Liu, Hao and Ye, Junze T and Blanchet, Jose and Si, Nian},
	year={2025},
	journal={Artificial Intelligence and Statistics (AISTATS)}
}
```
