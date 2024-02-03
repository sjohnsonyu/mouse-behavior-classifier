# mouse-behavior-classifier

## Environment Setup Instructions

### Step 1: Create a Conda Environment
Create a new Conda environment by running the following command in your terminal, replacing ENVNAME with your choice of name: `conda create -n ENVNAME python=3.8`

### Step 2: Activate the Environment
```conda activate ENVNAME```

### Step 3: Install PyTorch
Visit the official PyTorch website and find the installation command that matches your system configuration and CUDA version (if applicable).

Example for CPU-only:
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`

### Step 4: Install Other Dependencies
`pip install -r requirements.txt`

## Simple test run
Make dummy data by running: `python src/util/util.py`
X is a numpy array of shape `(n_samples, n_time_steps, n_features)`. X contains the *features* that are used for prediction.
Y is a numpy array of shape `(n_samples,)`. y contains the *labels* that tell whether the sample is from the control/treatment group.

Train the model by running: `python src/main.py --n_epochs 20`
Afterwards, the trained model should be in the `models/` directory and a plot of the training curve should be saved in `figures/` (both labeled with the timestamp).


## Running with real data

Get data into the shape `(n_samples, n_time_steps, n_features)`. Current code is written for X and y to be `.npy` files, but this can be changed. You may need to write a custom data-loading function to load in the data and get it in the right shape. Put your processed X and y files in the `data/` folder.

Importantly, there needs to be a held-out test set that you don't use for developing the model whatsoever and only use at the end of the project. Lmk if you want to take about data preprocessing!

Then run the model but pass in the flags that match the data parameters:
```
python3 src/main.py --n_keypoints YOUR_NUM_KEYPOINTS --x_filename YOUR_X_FILENAME --y_filename YOUR_Y_FILENAME
```



