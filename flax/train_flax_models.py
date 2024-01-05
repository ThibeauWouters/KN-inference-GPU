"""Example script to train a model with flax using the SVD method as usual."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import nmma as nmma

from nmma.em.training import SVDTrainingModel
from nmma.em.io import read_photometry_files
from nmma.em.utils import interpolate_nans

import inspect 
import nmma.em.model_parameters as model_parameters

import jax
import jaxlib
print("Checking CUDA")
print(jax.devices()) # check if CUDA is present

from flax import linen as nn  # Linen API
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses

# from clu import metrics
import optax

params = {"axes.grid": True,
        "text.usetex" : True,
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

plt.rcParams.update(params)

lcs_dir = "/home/urash/twouters/KN_Lightcurves/lightcurves/lcs_bulla_2022" # for remote SSH Potsdam
filenames = os.listdir(lcs_dir)
full_filenames = [os.path.join(lcs_dir, f) for f in filenames]
out_dir = "/home/urash/twouters/nmma_models/flax_models_new/" # initial flax models will be saved here


# Check if out directory exists, if not, create it
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
# If the directory exists, clean it
else:
    for file in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, file))


print("Loading data")
data = read_photometry_files(full_filenames)
data = interpolate_nans(data)

MODEL_FUNCTIONS = {
    k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
}

model_name = "Bu2022Ye"
model_function = MODEL_FUNCTIONS[model_name]
print("Extracting training data data")
training_data, parameters = model_function(data)

key = list(training_data.keys())[0]
example = training_data[key]
t = example["t"]
keys = list(example.keys())
filts = [k for k in keys if k not in parameters + ["t"]]

print("Filters used: ")
print(filts)

svd_ncoeff = 10
print("Getting SVD model now")
print(f"Will check for model at svd path: {out_dir}")
training_model = SVDTrainingModel(
        model_name,
        training_data,
        parameters,
        t,
        filts,
        n_coeff=svd_ncoeff,
        interpolation_type="flax",
        svd_path=out_dir # initial flax models will be saved here
        # start_training=False
    )


