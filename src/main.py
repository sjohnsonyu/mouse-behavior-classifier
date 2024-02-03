import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMClassifier

from util.util import get_dataloaders, plot_train_losses


def main(num_epochs,
         n_keypoints,
         batch_size,
         lr,
         num_layers,
         hidden_dim,
         data_path,
         x_filename,
         y_filename):
    exp_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(data_path, x_filename, y_filename, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss, and optimizer
    model = LSTMClassifier(n_keypoints * 3, hidden_dim, num_layers, 1).to(device)  # Assuming n_features = n_keypoints * 3
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader.dataset)
        epoch_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Validation step (optional, can be added here)


    model_path = f'models/{exp_name}_model.pth'
    torch.save(model.state_dict(), model_path)
    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total}%')

    # Plot the training losses
    plot_train_losses(epoch_losses, exp_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a LSTM model to classify what cohort mouse belongs to')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--n_keypoints', type=int, default=8, help='Number of keypoints')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--hidden_dim', type=int, default=50, help='Number of features in the hidden state')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to the data')
    parser.add_argument('--x_filename', type=str, default='X_dummy.npy', help='Filename for the input data')
    parser.add_argument('--y_filename', type=str, default='y_dummy.npy', help='Filename for the labels')
    args = parser.parse_args()

    main(num_epochs=args.n_epochs,
         n_keypoints=args.n_keypoints,
         batch_size=args.batch_size,
         lr=args.lr,
         num_layers=args.n_layers,
         hidden_dim=args.hidden_dim,
         data_path=args.data_path,
         x_filename=args.x_filename,
         y_filename=args.y_filename
         )
