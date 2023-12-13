import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from nmma.em.training import SVDTrainingModel
from nmma.em.model import SVDLightCurveModel
import nmma as nmma
import time
import arviz

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

from nmma.em.io import read_photometry_files
from nmma.em.utils import interpolate_nans

import inspect 
import nmma.em.model_parameters as model_parameters

MODEL_FUNCTIONS = {
    k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
}

model_name = "Bu2022Ye"
model_function = MODEL_FUNCTIONS[model_name]