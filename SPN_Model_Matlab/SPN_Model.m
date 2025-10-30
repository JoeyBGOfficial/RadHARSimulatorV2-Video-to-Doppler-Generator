%% SPN Model for Radar Image-Set-Based Human Activity Recognition
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Time: 2025.10.29.
% Platform: Matlab R2025b.
% Affiliation: Beijing Institute of Technology.
%
% Information:
%   This script implements a transfer learning approach using a custom architecture, 
%       inspired by SPGNet for activity recognition based on RadHARSimulator V2 generated image sets. 
%   It loads pre-trained network parameters, imports training and validation image data,
%       constructs the network layer graph with specialized SPGBlocks, 
%       and trains the model for multi-class classification (12 classes). 
%   The network is designed for efficiency on radar-derived images resized to 224x224x3. 
%   After training, it generates and saves plots for training/validation loss and accuracy curves, 
%       as well as a confusion matrix on the validation set.
%
% Input:
%   - Pre-trained network parameters from "SPN_Params.mat".
%   - Training and validation image datasets from the specified folder:
%       "\SPN_Model_Matlab\dataset" for combined data with random 80/20 split, or separate folders if using the commented option).
%   - No explicit function inputs; paths are hardcoded and may need adjustment.
%
% Output:
%   - net: The trained deep neural network for fall detection.
%   - traininfo: A structure containing training information, such as loss, accuracy, and validation metrics over epochs.
%   - Visualization figures: Saved as PNG files in the "visualizations" folder.
%
% Note:
%   - Ensure the Deep Learning Toolbox is installed in MATLAB.
%   - Data paths must be updated to match your local file system.
%   - The script supports two data import options: random split from a single folder or separate folders.
%   - Training uses ADAM optimizer with a learning rate of 0.00147 and may require GPU for faster execution.
%
% Reference:
%   - X. Wang, S. Lai, Z. Chai, X. Zhang and X. Qian, "SPGNet: Serial and Parallel Group Network," 
%       in IEEE Transactions on Multimedia, vol. 24, pp. 2804-2814, 2022.

%% Initialization of Matlab Script
clear all;
close all;
clc;
Font_Name = 'Palatino Linotype';                        % Font name used for plotting
Font_Fig_Basis = 16;                                    % Font size for all basic components and text of visualization.
Font_Title = 19;                                        % Font size for title of visualization.
Font_Axis =17;                                          % Font size for axis note of visualization.
JoeyBG_Colormap = [0.6196 0.0039 0.2588;                % Custom colormap for visualization
                   0.8353 0.2431 0.3098;
                   0.9569 0.4275 0.2627;
                   0.9922 0.6824 0.3804;
                   0.9961 0.8784 0.5451;
                   1.0000 1.0000 0.7490;
                   0.9020 0.9608 0.5961;
                   0.6706 0.8667 0.6431;
                   0.4000 0.7608 0.6471;
                   0.1961 0.5333 0.7412;
                   0.3686 0.3098 0.6353];
JoeyBG_Colormap_Flip = flip(JoeyBG_Colormap);           % Flipped version of the colormap
disp('---------- © Author: JoeyBG © ----------');

%% Load and Define Initial Parameters
% Load parameters for network initialization. 
% For transfer learning, the network initialization parameters are the parameters of the initial pretrained network.
trainingSetup = load("D:\JoeyBG_Research_Production\RadHARSimulator_V2\SPN_Model_Matlab\SPN_Params.mat");

%% Import Data
% Option 1: Import training and validation data from a single dataset and radomly splitted.
imdsTrain = imageDatastore("D:\JoeyBG_Research_Production\RadHARSimulator_V2\SPN_Model_Matlab\dataset", ...
    "IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8,"randomized");

% Option 2: Import training and testing data from two different folders.
% imdsTrain = imageDatastore("D:\JoeyBG_Research_Production\RadHARSimulator_V2\SPN_Model_Matlab\dataset", ...
%     "IncludeSubfolders",true,"LabelSource","foldernames");
% imdsValidation = imageDatastore("D:\JoeyBG_Research_Production\RadHARSimulator_V2\SPN_Model_Matlab\dataset", ...
%     "IncludeSubfolders",true,"LabelSource","foldernames");

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation);

%% Set Training Options
% Specify options to use when training.
opts = trainingOptions("adam",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.00147,...
    "LearnRateSchedule","piecewise",...
    "MaxEpochs",80,...
    "MiniBatchSize",64,...
    "OutputNetwork","best-validation-loss",...
    "Shuffle","every-epoch",...
    "ValidationFrequency",20,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

%% Create LayerGraph
% Create the layer graph variable to contain the network layers.
lgraph = layerGraph();

%% Add Layer Braches
% Add the branches of the network to the layer graph. Each branch is a linear array of layers.
tempLayers = [
    imageInputLayer([224 224 3],"Name","ImageInput")
    convolution2dLayer([3 3],12,"Name","Conv1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],24,"Name","Conv2","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","SPGBlock1 BatchNorm1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],24,"Name","SPGBlock1 Conv1","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv8","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv3","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv9","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv4","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv10","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv5","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv11","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv6","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv12","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv7","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv13","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv14","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv15","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv16","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv17","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv18","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],24,"Name","SPGBlock1 Conv19","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(12,"Name","SPGBlock1 DepthConcat1")
    convolution2dLayer([1 1],96,"Name","SPGBlock1 Conv20","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],96,"Name","SPGBlock1 Conv21","Padding","same","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","SPGBlock1 Addition1")
    batchNormalizationLayer("Name","SPGBlock2 BatchNorm1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],96,"Name","SPGBlock2 Conv1","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv8","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv3","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv9","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv4","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv10","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv5","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv11","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv6","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv12","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv7","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv13","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv14","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv15","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv16","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv17","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv18","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],96,"Name","SPGBlock2 Conv19","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(12,"Name","SPGBlock2 DepthConcat1")
    convolution2dLayer([1 1],192,"Name","SPGBlock2 Conv20","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],192,"Name","SPGBlock2 Conv21","Padding","same","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","SPGBlock2 Addition1")
    batchNormalizationLayer("Name","SPGBlock3 BatchNorm1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],192,"Name","SPGBlock3 Conv1","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv8","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv3","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv9","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv4","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv10","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv5","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv11","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv6","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv12","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv7","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv13","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv14","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv15","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv16","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv17","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv18","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],192,"Name","SPGBlock3 Conv19","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(12,"Name","SPGBlock3 DepthConcat1")
    convolution2dLayer([1 1],384,"Name","SPGBlock3 Conv20","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],384,"Name","SPGBlock3 Conv21","Padding","same","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","SPGBlock3 Addition1")
    convolution2dLayer([1 1],1024,"Name","Conv6","Padding","same")
    globalAveragePooling2dLayer("Name","GlobalPool1")
    fullyConnectedLayer(1024,"Name","FullyConnected1")
    fullyConnectedLayer(12,"Name","FullyConnected2")
    softmaxLayer("Name","Softmax1")
    classificationLayer("Name","ClassificationOutput")];
lgraph = addLayers(lgraph,tempLayers);

% Clean up helper variable.
clear tempLayers;

%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.
lgraph = connectLayers(lgraph,"SPGBlock1 BatchNorm1","SPGBlock1 Conv1");
lgraph = connectLayers(lgraph,"SPGBlock1 BatchNorm1","SPGBlock1 Conv21");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv1","SPGBlock1 Conv2");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv1","SPGBlock1 Conv3");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv1","SPGBlock1 Conv4");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv1","SPGBlock1 Conv5");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv1","SPGBlock1 Conv6");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv1","SPGBlock1 Conv7");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv8","SPGBlock1 Conv14");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv8","SPGBlock1 DepthConcat1/in1");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv9","SPGBlock1 Conv15");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv9","SPGBlock1 DepthConcat1/in2");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv10","SPGBlock1 Conv16");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv10","SPGBlock1 DepthConcat1/in3");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv11","SPGBlock1 Conv17");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv11","SPGBlock1 DepthConcat1/in4");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv12","SPGBlock1 Conv18");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv12","SPGBlock1 DepthConcat1/in5");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv13","SPGBlock1 Conv19");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv13","SPGBlock1 DepthConcat1/in6");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv14","SPGBlock1 DepthConcat1/in7");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv15","SPGBlock1 DepthConcat1/in8");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv16","SPGBlock1 DepthConcat1/in9");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv17","SPGBlock1 DepthConcat1/in10");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv18","SPGBlock1 DepthConcat1/in11");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv19","SPGBlock1 DepthConcat1/in12");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv20","SPGBlock1 Addition1/in2");
lgraph = connectLayers(lgraph,"SPGBlock1 Conv21","SPGBlock1 Addition1/in1");
lgraph = connectLayers(lgraph,"SPGBlock2 BatchNorm1","SPGBlock2 Conv1");
lgraph = connectLayers(lgraph,"SPGBlock2 BatchNorm1","SPGBlock2 Conv21");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv1","SPGBlock2 Conv2");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv1","SPGBlock2 Conv3");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv1","SPGBlock2 Conv4");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv1","SPGBlock2 Conv5");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv1","SPGBlock2 Conv6");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv1","SPGBlock2 Conv7");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv8","SPGBlock2 Conv14");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv8","SPGBlock2 DepthConcat1/in1");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv9","SPGBlock2 Conv15");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv9","SPGBlock2 DepthConcat1/in2");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv10","SPGBlock2 Conv16");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv10","SPGBlock2 DepthConcat1/in3");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv11","SPGBlock2 Conv17");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv11","SPGBlock2 DepthConcat1/in4");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv12","SPGBlock2 Conv18");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv12","SPGBlock2 DepthConcat1/in5");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv13","SPGBlock2 Conv19");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv13","SPGBlock2 DepthConcat1/in6");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv14","SPGBlock2 DepthConcat1/in7");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv15","SPGBlock2 DepthConcat1/in8");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv16","SPGBlock2 DepthConcat1/in9");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv17","SPGBlock2 DepthConcat1/in10");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv18","SPGBlock2 DepthConcat1/in11");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv19","SPGBlock2 DepthConcat1/in12");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv20","SPGBlock2 Addition1/in2");
lgraph = connectLayers(lgraph,"SPGBlock2 Conv21","SPGBlock2 Addition1/in1");
lgraph = connectLayers(lgraph,"SPGBlock3 BatchNorm1","SPGBlock3 Conv1");
lgraph = connectLayers(lgraph,"SPGBlock3 BatchNorm1","SPGBlock3 Conv21");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv1","SPGBlock3 Conv2");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv1","SPGBlock3 Conv3");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv1","SPGBlock3 Conv4");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv1","SPGBlock3 Conv5");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv1","SPGBlock3 Conv6");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv1","SPGBlock3 Conv7");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv8","SPGBlock3 Conv14");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv8","SPGBlock3 DepthConcat1/in1");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv9","SPGBlock3 Conv15");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv9","SPGBlock3 DepthConcat1/in2");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv10","SPGBlock3 Conv16");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv10","SPGBlock3 DepthConcat1/in3");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv11","SPGBlock3 Conv17");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv11","SPGBlock3 DepthConcat1/in4");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv12","SPGBlock3 Conv18");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv12","SPGBlock3 DepthConcat1/in5");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv13","SPGBlock3 Conv19");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv13","SPGBlock3 DepthConcat1/in6");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv14","SPGBlock3 DepthConcat1/in7");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv15","SPGBlock3 DepthConcat1/in8");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv16","SPGBlock3 DepthConcat1/in9");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv17","SPGBlock3 DepthConcat1/in10");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv18","SPGBlock3 DepthConcat1/in11");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv19","SPGBlock3 DepthConcat1/in12");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv20","SPGBlock3 Addition1/in2");
lgraph = connectLayers(lgraph,"SPGBlock3 Conv21","SPGBlock3 Addition1/in1");aa

%% Train Network
% Train the network using the specified options and training data.
analyzeNetwork(lgraph);
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

%% Plot Training and Validation Curves
% Extract epochs.
numIterations = numel(traininfo.TrainingLoss);
epochs = (1:numIterations) / opts.ValidationFrequency;  % Approximate epochs based on validation frequency.

% Plot Loss Curves.
figure(1);
plot(1:numIterations, traininfo.TrainingLoss, 'Color', ...
    JoeyBG_Colormap(2,:), 'LineWidth', 2, 'DisplayName', 'Training Loss');
hold on;
validationIterations = find(~isnan(traininfo.ValidationLoss));
plot(validationIterations, traininfo.ValidationLoss(validationIterations), 'Color', ...
    JoeyBG_Colormap_Flip(2,:), 'LineWidth', 2, 'DisplayName', 'Validation Loss');
set(gca, 'FontSize', Font_Fig_Basis, 'FontName', Font_Name);
xlabel('Iterations', 'FontSize', Font_Axis, 'FontName', Font_Name);
ylabel('Loss', 'FontSize', Font_Axis, 'FontName', Font_Name);
title('Training and Validation Loss Curves', 'FontSize', Font_Title, 'FontName', Font_Name);
legend('Location', 'best', 'FontSize', Font_Fig_Basis, 'FontName', Font_Name);
hold off;

% Plot Accuracy Curves.
figure(2);
plot(1:numIterations, traininfo.TrainingAccuracy, 'Color', ...
    JoeyBG_Colormap(2,:), 'LineWidth', 2, 'DisplayName', 'Training Accuracy');
hold on;
plot(validationIterations, traininfo.ValidationAccuracy(validationIterations), 'Color', ...
    JoeyBG_Colormap_Flip(2,:), 'LineWidth', 2, 'DisplayName', 'Validation Accuracy');
set(gca, 'FontSize', Font_Fig_Basis, 'FontName', Font_Name);
xlabel('Iterations', 'FontSize', Font_Axis, 'FontName', Font_Name);
ylabel('Accuracy (%)', 'FontSize', Font_Axis, 'FontName', Font_Name);
title('Training and Validation Accuracy Curves', 'FontSize', Font_Title, 'FontName', Font_Name);
legend('Location', 'best', 'FontSize', Font_Fig_Basis, 'FontName', Font_Name);
hold off;

%% Generate Confusion Matrix on Validation Set
% Perform inference on the validation set.
predictedLabels = classify(net, augimdsValidation);
trueLabels = imdsValidation.Labels;

% Plot the confusion matrix.
figure(3);
plotconfusion(trueLabels, predictedLabels);
set(findobj(gca, 'type', 'text'), 'FontSize', Font_Fig_Basis, 'FontName', Font_Name);
set(findall(gcf, '-property', 'FontSize'), 'FontSize', Font_Fig_Basis, 'FontName', Font_Name);
xlabel('Predicted Labels', 'FontSize', Font_Axis, 'FontName', Font_Name);
ylabel('True Labels', 'FontSize', Font_Axis, 'FontName', Font_Name);
title('Confusion Matrix', 'FontSize', Font_Title, 'FontName', Font_Name);
set(gcf, 'WindowState', 'maximized');

%% Save the Resulting Figures
for i = 1:3
    figure(i); % Activate the figure in sequence.
    filename = ['visualizations\Fig_' num2str(i) '.png']; % Define the name for saving.
    exportgraphics(gcf, filename, 'Resolution', 800);  % Export the image.
end