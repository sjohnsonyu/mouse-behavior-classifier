# mouse-behavior-classifier

## Simple test run
Make dummy data by running: `python3 src/util/util.py`
X is a numpy array of shape `(n_samples, n_time_steps, n_features)`. X contains the *features* that are used for prediction.
Y is a numpy array of shape `(n_samples,)`. y contains the *labels* that tell whether the sample is from the control/treatment group.

Train the model by running: `python3 src/main.py --n_epochs 20`
Afterwards, the trained model should be in the `models/` directory and a plot of the training curve should be saved in `figures/` (both labeled with the timestamp).


## Running with real data

Get data into the shape `(n_samples, n_time_steps, n_features)`. Current code is written for X and y to be `.npy` files, but this can be changed. You may need to write a custom data-loading function to load in the data and get it in the right shape. Put it in the `data/` folder.

Importantly, there needs to be a held-out test set that you don't use for developing the model whatsoever and only use at the end of the project. Lmk if you want to take about data preprocessing!

Then run the model but pass in the flags that match the data parameters:
```
python3 src/main.py --n_keypoints YOUR_NUM_KEYPOINTS --x_filename YOUR_X_FILENAME --y_filename YOUR_Y_FILENAME
```



