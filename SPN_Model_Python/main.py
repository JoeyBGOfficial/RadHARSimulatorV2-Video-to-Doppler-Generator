# -*- coding: utf-8 -*-
'''
Main Script
'''
# Former Author: Lerbron.
# Improved By: JoeyBG.
# Date: 2025.10.28.
# Platform: Python 3.10, paddlepaddle 3.0.0.
# Affiliation: Beijing Institute of Technology.
#
# Script Functionality:
#   - This script implements the complete training, validation, and visualization pipeline for the SPGNet model on a custom 12-class image classification dataset.
#   - The dataset is loaded from a folder structure, automatically split into training (80%) and validation (20%) sets, and augmented with resizing to 32x32 and ImageNet normalization.
#   - Training uses Label-Smoothed Cross-Entropy loss, Momentum optimizer with Cosine Annealing learning rate decay, and weight decay regularization.
#   - The best model based on validation accuracy is saved, along with the final model and optimizer states.
#   - Post-training visualization includes loss/accuracy curves, confusion matrix on the validation set using the best model, and random sample predictions with true/predicted labels.
#   - All outputs are saved under 'work/model' and 'work/visualizations' directories.
#
# Key Components:
#   - ImageDataset: Custom Dataset class using cv2 for image loading and Paddle transforms.
#   - LabelSmoothingCrossEntropy: Smoothed loss to prevent overconfidence.
#   - Training Loop: Epoch-wise training/validation with per-batch logging, accuracy tracking via paddle.metric.Accuracy, and scheduler step.
#   - Visualization Functions: plot_loss_curve, plot_acc_curve, plot_confusion_matrix, display_predictions.
#   - Model: SPGNet from models.py, configured for 12 classes, input resolution 32x32.
#
# Dataset Requirements:
#   - Folder structure: dataset/class_name/*.jpg/png (or other cv2-supported formats).
#   - Classes are inferred from sorted subfolder names; total classes must match num_classes=12.
#
# Usage:
#   - Place dataset in 'dataset/' directory.
#   - Run script directly; GPU is supported.
#   - Outputs saved automatically; no additional configuration needed beyond parameter section.

'''
Library Importation and Initialization
'''
# Import necessary libraries and set matplotlib to inline mode for displaying plots directly in the notebook.
# %matplotlib inline
import paddle
import numpy as np
import matplotlib.pyplot as plt
from paddle.vision.datasets import Cifar10
from paddle.vision.transforms import Compose, Normalize, Transpose, Resize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter
from paddle.io import Dataset, DataLoader
from paddle import nn
from paddle.nn import CrossEntropyLoss
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import itertools
import random
import math
import cv2 # For reasons that are unclear, network training does not work if the cv2 library is not introduced.
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from models import *

# Initialization of the Python script.
print("---------- Author: © JoeyBG © ----------")
# Execution the training with GPU of number 0.
# paddle.device.set_device('gpu:0')
# Excution the training with CPU.
paddle.device.set_device('cpu')
# Set the backend of the code running to cv2 image format.
paddle.vision.set_image_backend('cv2')

'''
Parameter Definition
'''
# Path definition.
data_dir = 'dataset' # Define the path to your dataset.
work_path = 'work/model' # Define the path for saving model.
visualization_path = 'work/visualizations' # Define the path for saving visualizations.

# Training parameters.
learning_rate = 0.00147 # Learning rate for training.
n_epochs = 20 # Number of epochs for training.
train_ratio = 0.8 # Ratio of training data to total dataset.
batch_size = 256 # Batch size predefined for training and validation dataloader.
num_classes = 12 # Number of classes in the dataset.
Estimation_Resolution = 32 # Define the resolution of input images. This parameter can be found in MATLAB feature augmentation script.
# paddle.seed(42) # Set random seed for reproducibility.
# np.random.seed(42)

'''
Data Augmentation and Normalization
'''
# Using data augmentation and normalization during training.
# train_tfm = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
# test_tfm = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
# Define the uniform transformations for both training and validation data augmentation.
transform = transforms.Compose([
    transforms.Resize((Estimation_Resolution, Estimation_Resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

'''
Dataset Construction
'''
# Define a custom dataset class.
class ImageDataset(paddle.io.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

# Read the training and validation dataset and split it into two.
dataset = []
label_list = sorted(os.listdir(data_dir))
for label_idx, label in enumerate(label_list):
    label_dir = os.path.join(data_dir, label)
    image_list = os.listdir(label_dir)
    random.shuffle(image_list)
    num_train = int(len(image_list) * train_ratio)
    dataset.extend([(os.path.join(label_dir, img), label_idx) for img in image_list])

# Create an instance of the custom dataset class.
full_dataset = ImageDataset(dataset, transform=transform)

# Calculate the number of samples for train and validation sets.
num_train_samples = int(len(full_dataset) * train_ratio)
num_val_samples = len(full_dataset) - num_train_samples

# Split the dataset into train and validation sets.
train_dataset, val_dataset = paddle.io.random_split(full_dataset, lengths=[num_train_samples, num_val_samples])

# Create data loaders for train and validation sets.
train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("Number of Datas in Training Set: %d" % len(train_dataset))
print("Number of Datas in Validation Set: %d" % len(val_dataset))

'''
Loss Function Definition
'''
# Definition of the label-smoothed cross-entropy loss function.
class LabelSmoothingCrossEntropy(nn.Layer):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):

        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(pred, axis=-1)
        idx = paddle.stack([paddle.arange(log_probs.shape[0], dtype=target.dtype), target], axis=1)
        nll_loss = paddle.gather_nd(-log_probs, index=idx)
        smooth_loss = paddle.mean(-log_probs, axis=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()
    
'''
Construction of the Models
'''
# Using standard SPGNet model construction with 12-class output.
model = SPGNet(num_classes=num_classes)
paddle.summary(model, (1, 3, 32, 32))

'''
Training and Validation
'''
# Create the directories if they do not exist.
if not os.path.exists(work_path):
    os.makedirs(work_path)
if not os.path.exists(visualization_path):
    os.makedirs(visualization_path)

# Plotting function definitions.
def plot_loss_curve(train_loss, val_loss, save_path):
    # Plots and saves the training and validation loss curves.
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', color='blue', linestyle='-')
    plt.plot(val_loss, label='Validation Loss', color='red', linestyle='--')
    plt.title('Loss Curve over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.show()
    plt.close()

def plot_acc_curve(train_acc, val_acc, save_path):
    # Plots and saves the training and validation accuracy curves.
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy', color='green', linestyle='-')
    plt.plot(val_acc, label='Validation Accuracy', color='orange', linestyle='--')
    plt.title('Accuracy Curve over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'accuracy_curve.png'))
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    # Plots and saves the confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Best Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.show()
    plt.close()

# Initialization.
criterion = LabelSmoothingCrossEntropy()
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=learning_rate, T_max=50000 // batch_size * n_epochs, verbose=False)
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, weight_decay=5e-4)
best_acc = 0.0

# Dictionaries to store history data for plotting.
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

# Main training loop.
for epoch in range(n_epochs):
    # Training phase.
    model.train()
    epoch_train_loss = 0.0
    accuracy_manager = paddle.metric.Accuracy()
    
    print(f"#=== Epoch: {epoch}, Learning Rate: {optimizer.get_lr():.10f} ===#")

    for batch_id, data in enumerate(train_loader):
        x_data, y_data = data
        labels = paddle.unsqueeze(y_data, axis=1)

        logits = model(x_data)
        loss = criterion(logits, y_data)
        
        # Calculate batch accuracy.
        acc = accuracy_manager.compute(logits, labels)
        accuracy_manager.update(acc)
        
        print(f"  Epoch: {epoch}, Batch: {batch_id + 1}/{len(train_loader)}, "
              f"Train Loss: {loss.item():.4f}, Train Acc: {acc.mean().item():.4f}")

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.clear_grad()
        epoch_train_loss += loss.item()

    # Record metrics for the current training epoch.
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_acc = accuracy_manager.accumulate()
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_acc)
    
    print(f"#=== Epoch: {epoch}, Avg Train Loss: {avg_train_loss:.4f}, Total Train Acc: {train_acc*100:.2f}% ===#")

    # Validation phase.
    model.eval()
    epoch_val_loss = 0.0
    val_accuracy_manager = paddle.metric.Accuracy()

    with paddle.no_grad():
        for batch_id, data in enumerate(val_loader):
            x_data, y_data = data
            labels = paddle.unsqueeze(y_data, axis=1)

            logits = model(x_data)
            loss = criterion(logits, y_data)
            acc = val_accuracy_manager.compute(logits, labels)
            val_accuracy_manager.update(acc)
            epoch_val_loss += loss.item()

    # Record metrics for the current validation epoch.
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_acc = val_accuracy_manager.accumulate()
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"#=== Epoch: {epoch}, Avg Val Loss: {avg_val_loss:.4f}, Total Val Acc: {val_acc*100:.2f}% ===#")

    if val_acc > best_acc:
        best_acc = val_acc
        paddle.save(model.state_dict(), os.path.join(work_path, 'best_model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(work_path, 'best_optimizer.pdopt'))
        print(f"Best model saved with accuracy: {best_acc*100:.2f}%\n")

# End of training and save the final model.
print(f"Training finished. Best validation accuracy: {best_acc*100:.2f}%")
paddle.save(model.state_dict(), os.path.join(work_path, 'final_model.pdparams'))
paddle.save(optimizer.state_dict(), os.path.join(work_path, 'final_optimizer.pdopt'))

# Start validation visualization.
print("Starting visualization...")

# Plot loss curves.
plot_loss_curve(history['train_loss'], history['val_loss'], visualization_path)
print(f"Loss curve saved to {os.path.join(visualization_path, 'loss_curve.png')}")

# Plot accuracy curves.
plot_acc_curve(history['train_acc'], history['val_acc'], visualization_path)
print(f"Accuracy curve saved to {os.path.join(visualization_path, 'accuracy_curve.png')}")

# Calculate and plot confusion matrix for the best model.
print("Calculating confusion matrix for the best model...")
# Load the best model's state dictionary.
best_model_path = os.path.join(work_path, 'best_model.pdparams')
model.set_state_dict(paddle.load(best_model_path))
model.eval()

# Get the predicted class indexes and names.
all_preds = []
all_labels = []
with paddle.no_grad():
    for data in val_loader:
        x_data, y_data = data
        logits = model(x_data)
        preds = paddle.argmax(logits, axis=1)       
        all_preds.extend(preds.numpy())
        all_labels.extend(y_data.numpy())
class_names = label_list

# Plot the confusion matrix.
plot_confusion_matrix(all_labels, all_preds, class_names, visualization_path)
print(f"Confusion matrix saved to {os.path.join(visualization_path, 'confusion_matrix.png')}")

# End of visualization.
print("Visualization finished.")

"""
Randomly Sample Testing & Visualizations
"""
# Define a function to display sample predictions from the validation set.
def display_predictions(model, val_dataset, class_names, num_images=10, save_path=None):
    # Set the model to evaluation mode.
    model.eval()
    
    # Select 10 random indices from the validation dataset.
    random_indices = random.sample(range(len(val_dataset)), num_images)
    
    # Create a 2 x 5 grid for displaying images.
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle('Sample Predictions from Validation Set', fontsize=16)

    for i, idx in enumerate(random_indices):
        # Get image and true label from the dataset by index.
        image, true_label_idx = val_dataset[idx]
        
        # Prepare the image tensor for the model: add a batch dimension.
        image_tensor = paddle.unsqueeze(image, axis=0) # The shape becomes [1, C, H, W].

        # Get model prediction.
        with paddle.no_grad():
            logits = model(image_tensor)
            pred_label_idx = paddle.argmax(logits, axis=1).item()
        true_label = class_names[true_label_idx]
        pred_label = class_names[pred_label_idx]
        ax = axes[i // 5, i % 5]
        
        # Display the image.
        # Note: If the image tensor is [C, H, W], we need to transpose it to [H, W, C] for imshow.
        # For grayscale [1, H, W], we squeeze it to [H, W].
        img_display = image.squeeze().numpy() if image.shape[0] == 1 else image.transpose([1, 2, 0]).numpy()
        ax.imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)

        # Set the title with prediction info, colored green for correct, red for incorrect.
        title_color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for the main title.
    if save_path:
        plt.savefig(os.path.join(save_path, 'sample_predictions.png'))
    plt.show()
    plt.close()

# Display sample predictions.
print("Displaying random sample predictions from the validation set...")
display_predictions(model, val_loader.dataset, class_names, num_images=10, save_path=visualization_path)
print(f"Sample predictions image saved to {os.path.join(visualization_path, 'sample_predictions.png')}")

# End of visualization.
print("Visualization finished.")