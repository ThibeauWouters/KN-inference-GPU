import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# NMMA imports
import nmma
from nmma.em.io import read_photometry_files
from nmma.em.utils import interpolate_nans
import inspect 
import nmma.em.model_parameters as model_parameters

MODEL_FUNCTIONS = {
    k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
}
MODEL_FUNCTIONS
from nmma.em.training import SVDTrainingModel

# Tensorflow imports 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler

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
        "figure.titlesize": 16
        }

plt.rcParams.update(params)


# Choose model and set location of the kilonova lightcurves
bulla_2022_dir = "/home/urash/twouters/KN_Lightcurves/lightcurves/lcs_bulla_2022"
bulla_2019_dir = "/home/urash/twouters/KN_Lightcurves/lightcurves/lcs_bulla_2019"

# Choose the model here
# model_name = "Bu2022Ye"
model_name = "Bu2019lm"
model_function = MODEL_FUNCTIONS[model_name]

# Set the location of the lightcurves and outdir based on chosen model
if model_name == "Bu2022Ye":
    lcs_dir = bulla_2022_dir
    
elif model_name == "Bu2019lm":
    lcs_dir = bulla_2019_dir

outdir = f"/home/urash/twouters/new_nmma_models/"

# Process the KN lightcurves
filenames = os.listdir(lcs_dir)
full_filenames = [os.path.join(lcs_dir, f) for f in filenames]
print(f"There are {len(full_filenames)} lightcurves for this model.")

print("Reading lightcurves and interpolating NaNs...")
data = read_photometry_files(full_filenames)
data = interpolate_nans(data)
keys = list(data.keys())
filts = sorted(list(set(data[keys[0]].keys()) - {"t"}))

print("Original filters:")
print(filts)

print("Reading lightcurves and interpolating NaNs... DONE")

# Limit to the filters of interest for the KN event that Peter is interested in:
if model_name == "Bu2022Ye":
    filts = ["ztfg", "ztfi", "ztfr"] # limited for now for Peter's KN event
else:
    filts = [f for f in filts if "sdss" in f]

print("Filters:")
print(filts)

# Get the time array
dat = pd.read_csv(full_filenames[0], delimiter=" ", escapechar='#')
dat = dat.rename(columns={" t[days]": "t"})
t = dat["t"].values

print("Genrating training data...")
training_data, parameters = model_function(data)
print("Genrating training data... DONE")

print("Getting SVD model")
svd_ncoeff = 10
training_model = SVDTrainingModel(
        model_name,
        training_data,
        parameters,
        t,
        filts,
        n_coeff=svd_ncoeff,
        interpolation_type="tensorflow",
        svd_path=outdir,
        start_training=False, # don't train, just prep the data
        load_model=False, # don't load a model, just prep the data
    )
print("Getting SVD model DONE")

print("Generating SVD model")
svd_model = training_model.generate_svd_model()
training_model.svd_model = svd_model
print("Generating SVD model DONE")

def lr_schedule(epoch, current_lr, nb_epochs=80, multiplier=0.25):
    if epoch == 0:
        return current_lr
    elif epoch % nb_epochs == 0:
        return current_lr * multiplier
    else:
        return current_lr
    
lr_scheduler = LearningRateScheduler(lr_schedule)

training_history_list = []

for i, f in enumerate(filts):
    
    print(f"============================================================================================================")
    print(f"Training {f} filter ({i+1}/{len(filts)})")

    X = training_model.svd_model[f]['param_array_postprocess'] # complete dataset of input data of network
    n_samples, input_ndim = X.shape
    print(f"Features (input) have shape {X.shape}")

    y = training_model.svd_model[f]['cAmat'].T
    _, output_ndim = y.shape
    print(f"Labels (output) have shape {y.shape}")
    
    # Perform train-validation split
    train_X, val_X, train_y, val_y = train_test_split(X, y, shuffle=True, test_size=0.1, random_state=0)
    
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
    
    model.compile(optimizer="adam", loss="mse")
    n_epochs = 200

    # fit the model
    training_history = model.fit(
        train_X,
        train_y,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(val_X, val_y),
        callbacks=[lr_scheduler], 
        verbose=True,
    )
    
    training_model.svd_model[f]["model"] = model
    training_history_list.append(training_history)
    
for filt, history in zip(filts, training_history_list):
    # Print the final loss values for training and validation
    print(f"Final training loss {filt}: {history.history['loss'][-1]}")
    print(f"Final validation loss {filt}: {history.history['val_loss'][-1]}")
        
    # Plot training & validation loss values
    plt.plot(history.history['loss'], label=f"{filt} training")
    plt.plot(history.history['val_loss'], label=f"{filt} validation")
    plt.yscale("log")
    plt.ylabel(f"Loss {filt}")
    plt.legend()
    plt.show()
    plt.close()