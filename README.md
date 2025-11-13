# RadHARSimulator V2: Video to Doppler Generator

## I. INTRODUCTION

<img width="2048" height="2048" alt="splash_screen" src="https://github.com/user-attachments/assets/b8fdd9e4-3276-45fe-bd61-953600910e79" />
Fig. 1. Splash screen of RadHARSimulator V1.

### Write Sth. Upfront:

The paper of this simulator can be found here: https://arxiv.org/abs/2511.09022.

This is the upgraded version of my previous work RadHARSimulator V1: https://github.com/JoeyBGOfficial/RadHARSimulatorV1-Model-Based-FMCW-Radar-Human-Activity-Recognition-Simulator. This work enables the direct simulation of free-space/through-the-wall radar echoes and maps from human activity captured in videos recorded by smartphones or cameras, employing a method that is not absolutely precise but somewhat effective. Additionally, similar to previous works, we simultaneously release an effective recognition neural network design with both MATLAB and Python implementation.

I would like to thank my mentors for the platform they have provided me. 

My software has not undergone extensive testing by a large number of users. There may still be areas for improvement during use. I welcome your valuable feedback and would be very grateful!

### Basic Information:

The V2 version of radar-based human activity recognition simulator (RadHARSimulator). This app presents an integrated pipeline for advanced human motion analysis from video and subsequent physics-based radar signature simulation. The process begins with computer vision techniques to extract detailed 3D human kinematics and culminates in the generation and analysis of corresponding radar data signatures.

My Email: JoeyBG@126.com;

Abstract: Radar-based human activity recognition (HAR) lies in providing decision-making insights for urban warfare and counter-terrorism through contactless perception, while safeguarding privacy and security in areas such as monitoring for elderly individuals living alone and smart home applications. However, this field still lacks a comprehensive simulation method for radar-based HAR. Existing software is developed based on models or motion-captured data, resulting in limited flexibility. To address this issue, a simulator that directly generates Doppler spectra from recorded video footage (RadHARSimulator V2) is presented in this paper. Both computer vision and radar modules are included in the simulator. In computer vision module, the real-time model for object detection with global nearest neighbor is first used to detect and track human targets in the video. Then, the high-resolution network is used to estimate two-dimensional poses of the detected human targets. Next, the three-dimensional poses of the detected human targets are obtained by nearest matching method. Finally, smooth temporal three-dimensional pose estimation is achieved through Kalman filtering. In radar module, pose interpolation and smoothing are first achieved through the Savitzky-Golay method. Second, the delay model and the mirror method are used to simulate echoes in both free-space and through-the-wall scenarios. Then, range-time map is generated using pulse compression, moving target indication, and DnCNN. Next, Doppler-time map (DTM) is generated using short-time Fourier transform and DnCNN again. Finally, the ridge features on the DTM are extracted using the maximum local energy method. In addition, a hybrid parallel-serial neural network architecture is proposed for radar-based HAR. Numerical experiments are conducted and analyzed to demonstrate the effectiveness of the designed simulator and the proposed network model.

Corresponding Papers:

[1] W. Gao, “RadHARSimulator V2: Video to Doppler Generator,” *arXiv (Cornell University)*, Nov. 2025, doi: 10.48550/arXiv.2511.09022.

### Notes:
**This app is currently for learning purposes only! Any commercial use is strictly prohibited. If you wish to use the app to generate data for paper publication, please cite our work. Appreciated!**

## II. HOW TO INSTALL

This app is packaged as a MATLAB toolbox. Follow the steps for installing.

1. Download the installer from GoogleDrive: https://drive.google.com/file/d/1F9DRkECVmazQp6NrIiTMsvBu8ICdDSUD/view?usp=sharing
2. Open MATLAB and double click the downloaded "RadHARSimulator_V2_Toolbox.mltbx" to install: <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/90640f77-f664-4d8b-8f78-44993c44e2d5" />
3. After the installation is completed, you can find the app in your MATLAB's APPs toolbar: <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f83cc779-d2c1-449c-99a8-7d12597945af" />
4. **[This step is essential!!!]** Open MATLAB "Add-On" Explorer, search and install two supported packages: "Computer Vision Toolbox Model for RTMDet Object Detection" and "Computer Vision Toolbox Model for Object Keypoint Detection": <img width="1918" height="1080" alt="image" src="https://github.com/user-attachments/assets/d56abf92-33e8-49c2-9b9b-a4d19ea75eda" /> <img width="1918" height="1080" alt="image" src="https://github.com/user-attachments/assets/48abaa4d-0517-4dc2-a756-15c3036beaec" />
5. After the above steps are done, the app will run normally. Feel free to use it!

## III. HOW TO USE

<img width="1651" height="847" alt="image" src="https://github.com/user-attachments/assets/6358d013-4a40-4e42-b2b7-df0b880295aa" />
Fig. 2. Software interface of the proposed simulator.<br><br>

After clicking the app botton in your MATLAB APPs toolbar, you will see the interface of the simulator. Hovering the mouse over the corresponding parameter box displays an explanation of the parameter to be entered. After choosing your video file and filling in the parameters, switch the “off” button to “on” to start the simulation. Upon simulation completion, the parameters, echo matrix, and various images will automatically stored to your desktop and pop up.

**Attention! While most videos recorded on mobile phones are in MP4 format, they are encoded using HEVC. MATLAB does not support reading this proprietary encoding format. Therefore, the software provides an online link for decoding HEVC MP4 files. Press the "HEVC Decoding" button.**

You can also access the aoo manual by tapping the "Instruction" button.

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

## V. SOME THINGS TO NOTE

**(1) Reproducibility Issues:** The network model receives image inputs in the form of three-channel RGB images.

**(2) Environment Issues:** The project consists of both MATLAB and Python code. The recommended MATLAB version is R2025a and above, and the recommended Python version is 3.10 and above. The program can be executed by both CPU and GPU environment.

**(3) Algorithm Design Issues:** This simulator is developed using the MATLAB toolbox, so the full version of MATLAB must be installed first. Additionally, the two add-on packages mentioned during the installation process are also required. Otherwise, the program will encounter errors during execution.

**(4) Right Issues: ⭐The project is limited to learning purposes only. Only part of the computer vision module utilizes MATLAB's pre-trained third-party models. Any use or interpretation without citation or authorized by me is not allowed!⭐**

Last but not least, hope that my work will bring positive contributions to the open source community in the filed of radar signal processing.

## VI. VERSION MANAGEMENT

| Version | Release Date | Supporting Package | Description |
| ----------- | ----------- | ----------- | ----------- |
| V2.0 | 2025.11.13 | MATLAB, Computer Vision Toolbox Model for RTMDet Object Detection, Computer Vision Toolbox Model for Object Keypoint Detection | The second version of the simulator. |
