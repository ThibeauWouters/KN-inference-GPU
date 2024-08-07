import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import jax.numpy as jnp
import pandas as pd
import copy
import matplotlib.pyplot as plt
# matlotlib settings
mpl_params = {"axes.grid": True,
        "text.usetex" : False, # TODO enable latex, but this breaks if filters have underscore
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}
plt.rcParams.update(mpl_params)

# NMMA imports
import nmma as nmma
from nmma.em.io import loadEvent

# flowMC imports
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

# jax imports
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
import jaxlib
import jax.numpy as jnp
import json 

from kim import Kim, Uniform, Composite

import sys
sys.path.append("../../../likelihood/")
import utils
import corner

print("Checking if CUDA is found:")
print(jax.devices())

### SETUP SCRIPT ###

n_chains = 1000
rng_key_set = initialize_rng_keys(n_chains, seed=42)
parameters = ['log10_mej_dyn', 'vej_dyn', 'Yedyn', 'log10_mej_wind', 'vej_wind', 'inclination_EM']
n_dim = len(parameters)

### PRIORS ###

log10_mej_dyn_prior = Uniform(-3, -1.7, naming="log10_mej_dyn")
vej_dyn_prior = Uniform(0.12, 0.25, naming="vej_dyn")
Yedyn_prior = Uniform(0.15, 0.3, naming="Yedyn")
log10_mej_wind_prior = Uniform(-2, -0.89, naming="log10_mej_wind")
vej_wind_prior = Uniform(0.03, 0.15, naming="vej_wind")
inclination_EM_prior = Uniform(0., np.pi/2., naming="inclination_EM")

prior_list = [log10_mej_dyn_prior, 
              vej_dyn_prior, 
              Yedyn_prior, 
              log10_mej_wind_prior, 
              vej_wind_prior, 
              inclination_EM_prior]

prior_range = [[prior.xmin, prior.xmax] for prior in prior_list]
prior = Composite(prior_list)

def posterior(theta, data):
    # NOTE: the data argument is unused?
    prior = utils.top_hat(theta, n_dim, prior_range)
    return utils.likelihood_fn_flax(theta) + prior

eps = 1e-5
mass_matrix = jnp.eye(n_dim)
# TODO tune it here
# mass_matrix = mass_matrix.at[0,0].set(1e-5)
# mass_matrix = mass_matrix.at[1,1].set(1e-4)
# mass_matrix = mass_matrix.at[2,2].set(1e-3)
# mass_matrix = mass_matrix.at[3,3].set(1e-3)
# mass_matrix = mass_matrix.at[7,7].set(1e-5)
# mass_matrix = mass_matrix.at[11,11].set(1e-2)
# mass_matrix = mass_matrix.at[12,12].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * eps}

outdir_name = "./outdir/"
kim = Kim(utils.likelihood_fn_flax,
          prior,
          n_loop_training=5,
          n_loop_production=5,
          n_local_steps=10,
          n_global_steps=10,
          n_chains=1000,
          n_epochs=50,
          learning_rate=0.001,
          max_samples=50000,
          momentum=0.9,
          batch_size=50000,
          use_global=True,
          keep_quantile=0.0,
          train_thinning=10,
          output_thinning=40,
          local_sampler_arg=local_sampler_arg,
          stopping_criterion_global_acc = 0.10,
          outdir_name=outdir_name
          )