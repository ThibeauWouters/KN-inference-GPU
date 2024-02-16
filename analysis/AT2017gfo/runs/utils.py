import copy
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import truncnorm
from scipy.interpolate import interp1d
import inspect 

from nmma.em.io import loadEvent
from nmma.em.utils import calc_lc_flax
from nmma.em.model import SVDLightCurveModel

import nmma.em.model_parameters as model_parameters

### PREAMBLE
MODEL_FUNCTIONS = {
    k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
}
MODEL_NAME = "Bu2022Ye"
model_function = MODEL_FUNCTIONS[MODEL_NAME]
MAG_NCOEFF = 10

def truncated_gaussian(m_det, m_err, m_est, lim):

    a, b = (-jnp.inf - m_est) / m_err, (lim - m_est) / m_err
    logpdf = truncnorm.logpdf(m_det, a, b, loc=m_est, scale=m_err)

    return logpdf

def get_chisq_filt(mag_abs, 
                   sample_times,
                   data_time, 
                   data_mag, 
                   data_sigma,
                   t0: float = 0.0,
                   error_budget: float = 1.0,
                   luminosity_distance = 44.0):
    
    """
    Function taken from nmma/em/likelihood.py and adapted to this case here
    
    This is a piece of the log likelihood function, which is the sum of the chisquare for a single filter, to decompose the likelihood calculation.
    """
    
    # TODO non-zero time
    # TODO include non-trivial error budget

    # Calculate apparent magnitude
    mag_app = mag_abs + 5.0 * jnp.log10(
        luminosity_distance * 1e6 / 10.0
    )
    
    # Limit to finite magnitudes
    # TODO implement check for finite
    sample_times_used = sample_times
    mag_app_used = mag_app
    
    # Add the error budget to the sigma
    data_sigma = jnp.sqrt(data_sigma ** 2 + error_budget ** 2)

    # Evaluate the light curve magnitude at the data points
    mag_est = jnp.interp(data_time, sample_times_used + t0, mag_app_used, left="extrapolate", right="extrapolate")

    # TODO get detection limit?
    detection_limit = jnp.inf
    minus_chisquare = jnp.sum(
        truncated_gaussian(
            data_mag,
            data_sigma,
            mag_est,
            detection_limit,
        )
    )
    
    return minus_chisquare

def top_hat(x, n_dim, prior_range):
    output = 0.
    for i in range(n_dim):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    return output

def log_likelihood(parameters, luminosity_distance = 44.0):
    """
    Function taken from nmma/em/likelihood.py and adapted to this case here
    
    TODO: 
    - separate LC params from parameters?
    - add error budget
    - add timeshift
    - add luminosity distance
    - this is assuming all data are "finite" and the LC is finite. Not checking this here since breaks JAX jit
    """
    
    # Generate the light curve
    _, _, mag_abs = calc_lc_function(parameters)
    
    minus_chisquare_total = 0.0
    for filt in filters:
        # Decompose the data of this filter
        data_time, data_mag, data_sigma  = copy.deepcopy(data[filt]).T
        mag_abs_filt = mag_abs[filt]
        # Compute the chi squared for this filter
        chisq_filt = get_chisq_filt(mag_abs_filt, sample_times, data_time, data_mag, data_sigma, luminosity_distance = luminosity_distance)
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

svd_path = "/home/urash/twouters/nmma_models/flax_models/"

lc_model = SVDLightCurveModel(
        MODEL_NAME,
        sample_times,
        svd_path=svd_path,
        parameter_conversion=None,
        mag_ncoeff=MAG_NCOEFF,
        lbol_ncoeff=None,
        interpolation_type="flax",
        model_parameters=None,
        filters=filters,
        local_only=True
)

# TODO not sure which function I want
calc_lc_given_params = lambda x: calc_lc_flax(sample_times, x, svd_mag_model=lc_model.svd_mag_model, svd_lbol_model=None, mag_ncoeff=MAG_NCOEFF, lbol_ncoeff=None, filters=filters)
calc_lc_given_params_jit = jax.jit(calc_lc_given_params)

calc_lc_function = calc_lc_given_params