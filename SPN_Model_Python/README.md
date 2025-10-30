## IV. ABOUT THE NETWORK MODEL

### A. Theory in Simple

For radar high-precision recognition tasks based on Doppler-Time Maps (DTM), this paper proposes a novel neural network architecture (Shown in Fig. 3). The key principle of this design lies in utilizing a serial, parallel multi-level deep concatenation fusion network architecture (DopSPN) to enable the perception of information across various scales and levels of complexity. It is equally applicable for recognition using DTM directly or using ridge feature maps extracted from DTM, achieving good validation accuracy.

Fig. 3. Structure of the proposed DopSPN model.

### B. Codes Explanation (Folder: SPN_Model_Matlab, SPN_Model_Python)

#### 1. SPN_Model.m
This MATLAB script implements a transfer learning approach using DopSPN architecture. It loads pre-trained network parameters, imports training and validation image datasets with random 80/20 split, constructs a deep network with specialized SPGBlocks featuring parallel grouped convolutions and depth concatenation, trains the model for 12-class classification using the Adam optimizer, and generates detailed visualizations including training and validation loss curves, accuracy curves, and a confusion matrix on the validation set.

**Input:** Pre-trained parameters from SPN_Params.mat and dataset directory containing subfolders for each activity class with radar-derived images.

**Output:** Trained network and training information structure, saved visualization figures for loss curves, accuracy curves, and confusion matrix.

#### 2. models.py
This Python script defines the DopSPN model architecture using the PaddlePaddle framework for efficient image classification. It implements core building blocks including ConvX for standard convolution with batch normalization, SqueezeExcitation for channel attention, and SPGLayer as the primary module with split-transform-merge strategy, parallel grouped convolutions, and residual connections. The main class assembles initial feature extraction layers, multiple SPGLayer stages with progressive downsampling, and a classification head with global average pooling and fully connected layers, supporting configurable versions for varying capacity.

**Input:** Input images of size 3×32×32 during model summary; configurable number of classes and version during instantiation.

**Output:** Instantiated model ready for training or inference, with detailed layer summary when executed directly.

#### 3. main.py
This Python script executes the complete training and evaluation pipeline for the DopSPN model on a 12-class radar activity recognition dataset using PaddlePaddle 3.0.0. It loads images from a structured folder, applies resizing and ImageNet normalization, splits data into training and validation sets, trains the model with label-smoothed cross-entropy loss and momentum optimizer under cosine annealing scheduling, tracks metrics per epoch, saves the best and final model states, and generates comprehensive visualizations including loss and accuracy curves, confusion matrix from the best model, and random sample predictions with true and predicted labels.

**Input:** Dataset directory with class-named subfolders containing radar images.

**Output:** Saved best and final model weights with optimizer states, console training logs, visualization images for loss curve, accuracy curve, confusion matrix, and sample predictions.

#### C. Datafiles Explanation (Folder: SPN_Model_Matlab)

#### 1. SPN_Params.mat

Predefined parameter file for the DopSPN model in MATLAB version.
