import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import time
import jax.numpy as jnp
import pandas as pd
import copy
import matplotlib.pyplot as plt
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
from nmma.em.likelihood import OpticalLightCurve

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
print("Checking if CUDA is found:")
print(jax.devices())

import utils
import json 
from tqdm import tqdm

################
### PREAMBLE ###
################

parameters = ['log10_mej_dyn', 'vej_dyn', 'Yedyn', 'log10_mej_wind', 'vej_wind', 'inclination_EM']
prior_range = jnp.array([[-3, 1.7], [0.12, 0.25], [0.15, 0.3], [-2, 0.89], [-2, -0.89], [0, jnp.pi / 2]])
n_dim = len(prior_range)

############
### BODY ###
############

start_time = time.time()

print("Loading the data")
cpu_posterior_file = "./data/AT2017gfo_original_result.json"
with open(cpu_posterior_file, "r") as f:
    cpu_posterior_json = json.load(f)
cpu_posterior = cpu_posterior_json["posterior"]["content"]

posterior_samples = np.array([cpu_posterior[param] for param in parameters])
posterior_samples = posterior_samples.T

log_likelihood_cpu = cpu_posterior["log_likelihood"]

# Get the maximum likelihood parameters
max_likelihood_idx = np.argmax(log_likelihood_cpu)
max_likelihood_params = posterior_samples[max_likelihood_idx]

# Get a dictionary of the max likelihood parameters as well
max_likelihood_params_dict = utils.array_to_dict(max_likelihood_params)
print("max_likelihood_params_dict")
print(max_likelihood_params_dict)

# Tensorflow model
print("Testing LC generation:")
lc_max_likelihood_tensorflow = utils.calc_lc_given_params_tensorflow(max_likelihood_params, utils.sample_times)
lc_max_likelihood_tensorflow_dict = {f: np.asarray(lc_max_likelihood_tensorflow[f]) for f in utils.filters}

# Flax model
lc_max_likelihood_flax = utils.calc_lc_given_params_flax(max_likelihood_params, utils.sample_times)
lc_max_likelihood_flax_dict = {f: np.asarray(lc_max_likelihood_flax[f]) for f in utils.filters}


print("Getting log L functions")
log_likelihood_value = utils.log_likelihood(max_likelihood_params, utils.sample_times, utils.data, utils.trigger_time)
print("log_likelihood_value flax")
print(log_likelihood_value)

likelihood_fn_flax = lambda x: utils.log_likelihood(x, utils.sample_times, utils.data, utils.trigger_time)

nmma_likelihood = OpticalLightCurve(utils.lc_model_tensorflow,
                                    utils.filters,
                                    utils.data,
                                    utils.trigger_time,
                                    )

nmma_likelihood.parameters = max_likelihood_params_dict
log_likelihood_nmma_value = nmma_likelihood.log_likelihood()
print("log_likelihood_nmma_value")
print(log_likelihood_nmma_value)

def likelihood_fn_nmma(x):
    nmma_likelihood.parameters = utils.array_to_dict(x)
    return nmma_likelihood.log_likelihood()

print("We got the likelihood functions")
end_time = time.time()
print("Time elapsed: ", end_time - start_time)

print("Jitting jax functions")

likelihood_fn_flax = jax.jit(likelihood_fn_flax)

# ---

N_samples = 1_000
prior_samples = np.random.uniform(prior_range[:, 0], prior_range[:, 1], size=(N_samples, n_dim))

# Evaluate both likelihoods at these points
print("Sampling flax now")
start = time.time()
# log_likelihoods_flax = np.array([likelihood_fn_flax(sample) for sample in tqdm(prior_samples)])
log_likelihoods_flax = jax.vmap(likelihood_fn_flax)(prior_samples)
end = time.time()

print("Time elapsed: ", end - start)

print("Sampling nmma now")
start = time.time()
log_likelihoods_nmma = np.array([likelihood_fn_nmma(sample) for sample in tqdm(prior_samples)])
end = time.time()

print("Time elapsed: ", end - start)

# Save to a npz file:
filename = "./likelihoods.npz"
np.savez(filename, prior_samples=prior_samples,log_likelihoods_flax=log_likelihoods_flax, log_likelihoods_nmma=log_likelihoods_nmma)