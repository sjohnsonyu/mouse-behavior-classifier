import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dummy_data(n_samples, n_time_steps, n_features, n_classes=2):
    """Make dummy data for a classification problem with different distributions for each class."""
    X = np.zeros((n_samples, n_time_steps, n_features))
    y = np.random.randint(0, n_classes, n_samples)
    
    # Define different distributions for each class
    for i in range(n_samples):
        if y[i] == 0:
            # Class 0: lower mean
            X[i] = np.random.rand(n_time_steps, n_features) * 50
        else:
            # Class 1: higher mean
            X[i] = np.random.rand(n_time_steps, n_features) * 50 + 20

    return X, y


def get_dataloaders(data_path, x_filename, y_filename, batch_size):
    X = np.load(data_path + x_filename)
    y = np.load(data_path + y_filename)

    # Split data into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)  # 70% train, 30% for val+test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)  # Split the 30% into 15% val, 15% test

    # Convert to PyTorch tensors
    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def plot_train_losses(epoch_losses, exp_name):
    """Plot the training losses over epochs."""
    plt.plot(epoch_losses, label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'figures/{exp_name}_train_losses.png')


if __name__ == '__main__':
    np.random.seed(0)
    
    n_samples = 100
    n_time_steps = 10 * 60 * 5 * 2
    n_keypoints = 8
    n_features = n_keypoints * 3
    n_classes = 2

    # Make dummy data
    print('Making dummy data...')
    X, y = make_dummy_data(n_samples, n_time_steps, n_features, n_classes)
    print(X.shape, y.shape)

    # Save X and y as numpy arrays
    np.save('data/X_dummy.npy', X)
    np.save('data/y_dummy.npy', y)
