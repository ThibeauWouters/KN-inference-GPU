### TODO: add this if the inference looks good, take over from old code


# import numpy as np 
# import matplotlib.pyplot as plt
# import json
# import corner 

# params = {"axes.grid": True,
#         "text.usetex" : True,
#         "font.family" : "serif",
#         "ytick.color" : "black",
#         "xtick.color" : "black",
#         "axes.labelcolor" : "black",
#         "axes.edgecolor" : "black",
#         "font.serif" : ["Computer Modern Serif"],
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#         "axes.labelsize": 16,
#         "legend.fontsize": 16,
#         "legend.title_fontsize": 16,
#         "figure.titlesize": 16}

# plt.rcParams.update(params)

# # Improved corner kwargs
# default_corner_kwargs = dict(bins=40, 
#                         smooth=1., 
#                         show_titles=False,
#                         label_kwargs=dict(fontsize=16),
#                         title_kwargs=dict(fontsize=16), 
#                         color="blue",
#                         # quantiles=[],
#                         # levels=[0.9],
#                         plot_density=True, 
#                         plot_datapoints=False, 
#                         fill_contours=True,
#                         max_n_ticks=4, 
#                         min_n_ticks=3,
#                         save=False)

# # parameter_names = ['inclination_EM', 'log10_mej_dyn', 'vej_dyn', 'Yedyn', 'log10_mej_wind', 'vej_wind', 'timeshift', 'luminosity_distance', 'Ebv']
# parameter_names = ['log10_mej_dyn', 'vej_dyn', 'log10_mej_wind', 'vej_wind', 'Yedyn', 'inclination_EM']
# labels = ['$\\log_{10}M^{\\rm{dyn}}_{\\rm{ej}}$', '$V^{\\rm{dyn}}_{\\rm{ej}}$', '$\\log_{10}M^{\\rm{wind}}_{\\rm{ej}}$', '$V^{\\rm{wind}}_{\\rm{ej}}$', '$Y_{\\rm{e}}$', '$\\iota$']
# colors = ["blue", "orange", "red", "green", "purple", "yellow", "cyan", "magenta"]

# def get_posterior_from_filename(results_filename,
#                                 return_likelihood = False):
#     with open(results_filename, "r") as f:
#         results = json.load(f)
#     posterior = results["posterior"]["content"]

#     if return_likelihood:
#         log_likelihood = posterior['log_likelihood']    

#         return posterior, log_likelihood
#     else:
#         return posterior
    
    
# def get_samples(posterior):
#     posterior_dict = {}
#     for param in parameter_names:
#         posterior_dict[param] = posterior[param]
#     posterior_samples = np.array([posterior_dict[param] for param in parameter_names]).T
    
#     return posterior_samples