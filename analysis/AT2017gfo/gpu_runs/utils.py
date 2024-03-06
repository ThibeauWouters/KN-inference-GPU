import copy
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import truncnorm
from scipy.interpolate import interp1d
import inspect 
from nmma.em.io import loadEvent
from nmma.em.utils import calc_lc_flax, calc_lc, getFilteredMag
from nmma.em.model import SVDLightCurveModel
import nmma.em.model_parameters as model_parameters

from jaxtyping import Array

################
### PREAMBLE ###
################

MODEL_FUNCTIONS = {
    k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
}
MODEL_NAME = "Bu2022Ye"
model_function = MODEL_FUNCTIONS[MODEL_NAME]
MAG_NCOEFF = 10

def truncated_gaussian(m_det, m_err, m_est, upper_lim, lower_lim = -9999.0):

    a, b = (lower_lim - m_est) / m_err, (upper_lim - m_est) / m_err
    logpdf = truncnorm.logpdf(m_det, a, b, loc=m_est, scale=m_err)

    return logpdf

def get_chisq_filt(mag_app, 
                   sample_times,
                   data_time, 
                   data_mag, 
                   data_sigma,
                   t0: float = 0.0,
                   error_budget: float = 1.0,
                   upper_lim: float = 9999.0, 
                   lower_lim: float = -9999.0
                   ):
    
    """
    Function taken from nmma/em/likelihood.py and adapted to this case here
    
    This is a piece of the log likelihood function, which is the sum of the chisquare for a single filter, to decompose the likelihood calculation.
    """
    
    # Add the error budget to the sigma
    data_sigma = jnp.sqrt(data_sigma ** 2 + error_budget ** 2)

    # Evaluate the light curve magnitude at the data time points
    mag_est = jnp.interp(data_time, sample_times + t0, mag_app, left="extrapolate", right="extrapolate")

    minus_chisquare = jnp.sum(
        truncated_gaussian(
            data_mag,
            data_sigma,
            mag_est,
            upper_lim=upper_lim,
            lower_lim=lower_lim,
        )
    )
    
    return minus_chisquare

def top_hat(x, n_dim, prior_range):
    output = 0.
    for i in range(n_dim):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    return output

def log_likelihood(parameters, 
                   times,
                   data,
                   t0, 
                   luminosity_distance = 44.0):
    """
    Function taken from nmma/em/likelihood.py and adapted to this case here
    
    TODO: 
    - separate LC params from parameters?
    - add error budget
    - add timeshift
    - add luminosity distance
    - this is assuming all data are "finite" and the LC is finite. Not checking this here since breaks JAX jit
    """
    
    # Generate the light curve, this returns the apparent magnitude
    mag_app_dict = calc_lc_given_params_flax(parameters, times)
    
    minus_chisquare_total = 0.0
    filters = list(data.keys())
    for filt in filters:
        # Decompose the data of this filter
        data_time, data_mag, data_sigma  = copy.deepcopy(data[filt]).T
        mag_est_filt = mag_app_dict[filt]
        # Compute the chi squared for this filter
        chisq_filt = get_chisq_filt(mag_est_filt, sample_times, data_time, data_mag, data_sigma, t0=t0, luminosity_distance = luminosity_distance)
        minus_chisquare_total += chisq_filt

    log_prob = minus_chisquare_total

    return log_prob


### INITIALIZATION

data_file = "../data/AT2017gfo_no_inf.dat"
trigger_time = 57982.5285236896
tmin, tmax = 0.05, 14
data = loadEvent(data_file)
filters = list(data.keys())

sample_times = jnp.linspace(tmin, tmax, 1_000)
sample_times_np = np.asarray(sample_times)

### Lightcurve models

# flax model
svd_path_flax = "/home/urash/twouters/nmma_models/flax_models/"
lc_model_flax = SVDLightCurveModel(
        MODEL_NAME,
        sample_times,
        svd_path=svd_path_flax,
        parameter_conversion=None,
        mag_ncoeff=MAG_NCOEFF,
        lbol_ncoeff=None,
        interpolation_type="flax",
        model_parameters=None,
        filters=filters,
        local_only=True
)

def calc_lc_given_params_flax(params: Array, 
                              times: Array, 
                              luminosity_distance: float = 44.0) -> Array:
    
    result_dict = {}
    
    for filt in filters:
        # Compute the LC
        _, _, mag_abs = calc_lc_flax(times, params, svd_mag_model=lc_model_flax.svd_mag_model, svd_lbol_model=None, mag_ncoeff=MAG_NCOEFF, lbol_ncoeff=None, filters=filters)
        
        # Convert to apparent magnitude
        mag_abs_filt = getFilteredMag(mag_abs, filt)
        mag_app_filt = mag_abs_filt + 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)
        
        # Save
        result_dict[filt] = mag_app_filt
        
    return result_dict

# Tensorflow model

svd_path_peter = "/home/enlil/ppang/Projects/AT2017gfo_chemical_tf/inference/tensorflow_model/"
lc_model_tensorflow = SVDLightCurveModel(
        MODEL_NAME,
        sample_times,
        svd_path=svd_path_peter,
        parameter_conversion=None,
        mag_ncoeff=MAG_NCOEFF,
        lbol_ncoeff=None,
        interpolation_type="tensorflow",
        model_parameters=None,
        filters=filters,
        local_only=True
)

def calc_lc_given_params_tensorflow(params: np.array, 
                                    times: np.array, 
                                    luminosity_distance: float = 44.0) -> np.array:
    
    result_dict = {}
    
    for filt in filters:
        # Compute the LC
        _, _, mag_abs = calc_lc(times, params, svd_mag_model=lc_model_tensorflow.svd_mag_model, svd_lbol_model=None, mag_ncoeff=MAG_NCOEFF, lbol_ncoeff=None, filters=filters, interpolation_type="tensorflow")
        
        # Convert to apparent magnitude
        mag_abs_filt = getFilteredMag(mag_abs, filt)
        mag_app_filt = mag_abs_filt + 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)
        
        # Save
        result_dict[filt] = mag_app_filt
        
    return result_dict