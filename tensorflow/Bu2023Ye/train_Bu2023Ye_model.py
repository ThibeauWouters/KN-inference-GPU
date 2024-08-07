import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
import tqdm
import inspect 

# NMMA imports
import nmma
from nmma.em.io import read_photometry_files
from nmma.em.utils import interpolate_nans
from nmma.em.training import SVDTrainingModel
import nmma.em.model_parameters as model_parameters

# tensorflow imports
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

print("CUDA there?")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

params = {"axes.grid": True,
        "text.usetex" : False,
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
        "figure.titlesize": 16
        }

plt.rcParams.update(params)

# lcs_dir = "/home/urash/twouters/KN_lightcurves/lightcurves/bulla_2023" # location on the Potsdam cluster

# This is the new model with the updated heat curves
lcs_dir = "/home/urash/twouters/KN_lightcurves/lightcurves/bulla_2023_newheat/possis_newheat_lcs" # location on the Potsdam cluster
model_name = "Bu2023Ye"

filenames = os.listdir(lcs_dir)
full_filenames = [os.path.join(lcs_dir, f) for f in filenames]
print(f"There are {len(full_filenames)} lightcurves for this model.")

dat = pd.read_csv(full_filenames[0], delimiter=" ", escapechar='#')
dat = dat.rename(columns={" t[days]": "t"})
# dat.head() # show first few rows
t = dat["t"].values

value_columns = dat.columns[1:-1] # discard first and last, to get "true" data columns

print("Interpolating data")
data = read_photometry_files(full_filenames)
data = interpolate_nans(data)
keys = list(data.keys())
filts = sorted(list(set(data[keys[0]].keys()) - {"t"}))

# These are the filters that Shreya is training on:
filts = ["sdssu", "ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y", "2massj", "2massh", "2massks"]

print("filts")
print(filts)

MODEL_FUNCTIONS = {
    k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
}

print("Getting parameters")
model_function = MODEL_FUNCTIONS[model_name]
training_data, parameters = model_function(data)

## There are two file name conventions here, so we have to check 
all_keys = list(training_data.keys())
first_file_names = [f for f in all_keys if f.startswith("MDyn")]
second_file_names = [f for f in all_keys if f.startswith("dyn")]

# for example_key in [first_file_names[0], second_file_names[0]]:
#     print("------------------")
#     example_training_data = training_data[example_key]
#     print(f"Keys (and some values) of dictionary of {example_key}")
#     for key in example_training_data.keys():
#         value = example_training_data[key]
#         if key in parameters:
#             print(f"{key} : {value}")

svd_ncoeff = 10
svd_path = "/home/urash/twouters/new_nmma_models/newheat/Bu2023_tf/"
training_model = SVDTrainingModel(
        model_name,
        training_data,
        parameters,
        t,
        filts,
        n_coeff=svd_ncoeff,
        interpolation_type="tensorflow",
        svd_path = svd_path,
        start_training=False, # don't train, just prep the data
        load_model = False # don't load the model
    )

svd_model = training_model.generate_svd_model()
training_model.svd_model = svd_model

### TRAINING

print("Training the model...")

show_loss_curves = True

for i, filt in enumerate(filts):
    
    print(f"Training for {filt} filter... ({i+1}/{len(filts)})")

    X = training_model.svd_model[filt]['param_array_postprocess']
    n_samples, input_ndim = X.shape

    y = training_model.svd_model[filt]['cAmat'].T
    _, output_ndim = y.shape

    # Do the train validation split
    train_X, val_X, train_y, val_y = train_test_split(X, y, shuffle=True, test_size=0.25, random_state=0)
    
    # Build up the NN architecture
    # TODO this is chosen randomly, can improve the architecture?
    model = Sequential()
    model.add(
        Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(train_X.shape[1],),
        )
    )
    model.add(
        Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(train_X.shape[1],),
        )
    )
    model.add(
        Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(train_X.shape[1],),
        )
    )
    model.add(Dense(training_model.n_coeff))

    # # Show the architecture:
    # model.summary()

    # Compile the model and fit it
    model.compile(optimizer="adam", loss="mse")
    n_epochs = 100
    training_history = model.fit(
        train_X,
        train_y,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(val_X, val_y),
        verbose=True,
    )

    if show_loss_curves:
        train_loss = training_history.history["loss"]
        val_loss = training_history.history["val_loss"]
        plt.figure(figsize=(12, 5))
        plt.plot([i+1 for i in range(len(train_loss))], train_loss, '-o', color="blue", label="Training loss")
        plt.plot([i+1 for i in range(len(val_loss))], val_loss, '-o', color="red", label="Validation loss")
        plt.legend()
        plt.xlabel("Training epoch")
        plt.ylabel("MSE")
        plt.yscale('log')
        plt.title(filt)
        plt.savefig(f"./figures/{model_name}_{filt}_loss.png")
        plt.close()
        
    # Also save the model as attribute to the object, see NMMA source code for this
    training_model.svd_model[filt]["model"] = model
    
training_model.svd_path = svd_path

training_model.save_model()