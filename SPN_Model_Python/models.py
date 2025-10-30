# -*- coding: utf-8 -*-
'''
Model Construction Script
'''
# Former Author: Lerbron.
# Improved By: JoeyBG.
# Date: 2025.10.28.
# Platform: Python 3.10, paddlepaddle 3.0.0.
# Affiliation: Beijing Institute of Technology.
#
# Network Structure:
#   - The improved SPGNet model is a convolutional neural network designed for efficient image classification. 
#       It begins with a 'first_conv' module, a sequence of two ConvX blocks, to perform initial feature extraction. 
#       The main body of the network is constructed from multiple stages of SPGLayer blocks. 
#       Each SPGLayer employs a split-transform-merge strategy, where input features are first projected by a 1x1 convolution, 
#       then processed through two parallel paths of grouped convolutions, and finally concatenated. 
#       An optional Squeeze-and-Excitation (SE) block can be applied to the concatenated features for channel-wise attention. 
#       A residual skip connection is used to facilitate gradient flow. 
#       The network progressively downsamples feature maps between stages to build a hierarchical representation. 
#       The final classification head consists of a 1x1 convolution, followed by global average pooling and fully connected layers, 
#       outputting the class scores.
#
# Modules Description:
#   - ConvX: A fundamental convolutional block consisting of a Conv2D layer, a BatchNorm2D layer, and an optional ReLU activation.
#   - SqueezeExcitation: An attention mechanism that adaptively recalibrates channel-wise feature responses.
#   - SPGLayer: The core building block of the network, featuring a multi-path structure with grouped convolutions and a residual connection.
#   - SPGNet: The main network class that assembles the complete model architecture from the constituent blocks.
#   - AdaptiveAvgPool2D: A global average pooling layer that reduces each feature map to a single value.
#   - Linear: Fully connected layers used for the final classification.
#
# Usage:
#   - This script can be used to define and instantiate the improved SPGNet model for image classification tasks.
#   - The model architecture can be configured through different versions specified.
#
# Dataset Used:
#   - This model is a general-purpose backbone for image classification and can be trained on standard folder datasets.
#   - The user should ensure the input image size and the number of classes are set appropriately for their specific dataset.

# Import necessary libraries.
import paddle
from paddle import nn
import paddle.nn.functional as F

# A standard convolutional block.
class ConvX(nn.Layer):
    """
    A basic building block composed of Conv2D, BatchNorm2D, and an optional activation.
    """
    def __init__(self, in_planes, out_planes, groups, kernel_size=3, stride=1, act="relu"):
        """
        Initializes the ConvX layer.
        """
        super(ConvX, self).__init__()
        # 2D convolutional layer.
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias_attr=False)
        # Batch normalization layer.
        self.bn = nn.BatchNorm2D(out_planes)

        # Activation function.
        self.act = None
        if act == "relu":
            self.act = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass for the ConvX block.
        """
        out = self.bn(self.conv(x))
        if self.act != None:
            out = self.act(out)
        return out

# Squeeze-and-Excitation block.
class SqueezeExcitation(nn.Layer):
    """
    Implements the Squeeze-and-Excitation (SE) block for channel-wise attention.
    """
    def __init__(self, inplanes, se_ratio=0.25):
        """
        Initializes the SqueezeExcitation block.
        """
        super(SqueezeExcitation, self).__init__()
        hidden_dim = int(inplanes * se_ratio)
        # Squeeze operation: Global Average Pooling.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation operation: two fully connected layers.
        self.conv1 = nn.Conv2D(in_channels=inplanes, out_channels=hidden_dim, kernel_size=1, bias_attr=False)
        self.conv2 = nn.Conv2D(in_channels=hidden_dim, out_channels=inplanes, kernel_size=1, bias_attr=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass for the SE block.
        """
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # Apply sigmoid to get channel weights.
        out = F.sigmoid(out)
        # Recalibrate original feature map.
        return x * out

# SPGNet layer structure.
class SPGLayer(nn.Layer):
    """
    The main building block of SPGNet, featuring a Split-Point-Group architecture.
    """
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, reduce_num=2, group_num=8, use_se=False):
        """
        Initializes the SPGLayer.
        """
        super(SPGLayer, self).__init__()
        self.use_se = use_se
        # Optional Squeeze-and-Excitation block.
        if self.use_se:
            self.se = SqueezeExcitation(out_planes * 2)

        # Input convolution to reduce channel dimension.
        self.conv_in = ConvX(in_planes, int(out_planes * 0.5), groups=1, kernel_size=1, stride=1, act="relu")
        # Two parallel grouped convolution paths.
        self.conv1 = ConvX(int(out_planes * 0.5), int(out_planes * 0.5), groups=group_num, kernel_size=kernel, stride=stride, act="relu")
        self.conv2 = ConvX(int(out_planes * 0.5), int(out_planes * 0.5), groups=group_num, kernel_size=kernel, stride=1, act="relu")
        # Output convolution to merge features.
        self.conv_out = ConvX(int(out_planes*1.0), out_planes, groups=1, kernel_size=1, stride=1, act=None)

        self.act = nn.ReLU()
        
        self.stride = stride
        self.skip = None
        # Skip connection to handle dimension and stride changes.
        if stride == 1 and in_planes != out_planes:
            # Handle channel mismatch with a 1x1 convolution.
            self.skip = nn.Sequential(
                nn.Conv2D(in_planes, out_planes, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(out_planes)
            )

        if stride == 2 and in_planes != out_planes:
            # Handle downsampling and channel mismatch.
            self.skip = nn.Sequential(
                nn.Conv2D(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias_attr=False),
                nn.BatchNorm2D(in_planes),
                nn.Conv2D(in_planes, out_planes, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(out_planes)
            )

    def forward(self, x):
        """
        Defines the forward pass for the SPGLayer.
        """
        skip = x
        # Initial 1x1 convolution.
        out = self.conv_in(x)
        # First convolutional path.
        out1 = self.conv1(out)
        # Second convolutional path.
        out2 = self.conv2(out1)
        # Concatenate features from the two paths.
        out_cat = paddle.concat((out1, out2), axis=1)

        # Apply SE block if enabled.
        if self.use_se:
            out_cat = self.se(out_cat)
        # Final 1x1 convolution to merge features.
        out = self.conv_out(out_cat)

        # Apply skip connection if defined.
        if self.skip is not None:
            skip = self.skip(skip)
        # Add skip connection (residual).
        out += skip
        return self.act(out)

# Main structure of SPGNet.
class SPGNet(nn.Layer):
    """
    The main SPGNet model for image classification.
    """
    # Configuration for different versions of SPGNet.
    # Format: [initial_channels_1, initial_channels_2, (out_planes, num_blocks, stride, group_num, kernel), ...]
    cfgs = {
        "s2p6": [12, 24, 
            (96 , 4, 2, 6, 3),
            (192, 8, 2, 6, 3),
            (384, 3, 2, 6, 3)],
        "s2p7": [14, 28, 
            (112, 4, 2, 7, 3),
            (224, 8, 2, 7, 3),
            (448, 3, 2, 7, 3)],
        "s2p8": [16, 32, 
            (128, 4, 2, 8, 3),
            (256, 8, 2, 8, 3),
            (512, 3, 2, 8, 3)],
        "s2p9": [18, 36, 
            (144, 4, 2, 9, 3),
            (288, 8, 2, 9, 3),
            (576, 3, 2, 9, 3)]
    }

    def __init__(self, num_classes=1000, dropout=0.2, version="s2p6"):
        """
        Initializes the SPGNet model.
        """
        super(SPGNet, self).__init__()
        cfg = self.cfgs[version]

        # Initial convolutional layers.
        self.first_conv = nn.Sequential(
            ConvX(3 , cfg[0], 1, 3, 1, "relu"),
            ConvX(cfg[0], cfg[1], 1, 3, 1, "relu")
        )

        # Stack of SPGLayers.
        self.layers = self._make_layers(in_planes=cfg[1], cfg=cfg[2:])

        # Final convolutional layer before pooling.
        self.conv_last = ConvX(cfg[4][0], 1024, 1, 1, 1, "relu")

        # Classification head.
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(1024, 1024, bias_attr=False)
        self.bn = nn.BatchNorm1D(1024)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(1024, num_classes, bias_attr=False)
        # Initialize model weights.
        self.init_params()

    def init_params(self):
        """
        Initializes the weights of the network with specific schemes.
        """
        for name, m in self.named_sublayers():
            if isinstance(m, nn.Conv2D):
                if 'first' in name:
                    n = nn.initializer.Normal(std=.01)
                    n(m.weight)
                else:
                    n = nn.initializer.Normal(std=1.0 / m.weight.shape[1])
                    n(m.weight)
                if m.bias is not None:
                    zero = nn.initializer.Constant(0.)
                    zero(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                # Special initialization for the last batch norm in SPGLayer.
                if name.endswith("conv_out.bn.weight"):
                    zero = nn.initializer.Constant(0.)
                    zero(m.weight)
                else:
                    one = nn.initializer.Constant(1.)
                    one(m.weight)
                if m.bias is not None:
                    c = nn.initializer.Constant(0.0001)
                    c(m.bias)
                zero = nn.initializer.Constant(0.)
                zero(m._mean)
            elif isinstance(m, nn.BatchNorm1D):
                one = nn.initializer.Constant(1.)
                one(m.weight)
                if m.bias is not None:
                    c = nn.initializer.Constant(0.0001)
                    c(m.bias)
                zero = nn.initializer.Constant(0.)
                zero(m._mean)
            elif isinstance(m, nn.Linear):
                n = nn.initializer.Normal(std=.01)
                n(m.weight)
                if m.bias is not None:
                    zero = nn.initializer.Constant(0.)
                    zero(m.bias)

    def _make_layers(self, in_planes, cfg):
        """
        Builds the main layers of the network by stacking SPGLayers.
        """
        layers = []
        for out_planes, num_blocks, stride, group_num, kernel in cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(SPGLayer(in_planes, out_planes, kernel, stride, reduce_num=2, group_num=group_num))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the complete forward pass of the SPGNet model.
        """
        # Initial feature extraction.
        out = self.first_conv(x)
        # Main SPGLayer blocks.
        out = self.layers(out)
        # Final convolution.
        out = self.conv_last(out)
        # Global pooling and classification.
        out = self.gap(out).flatten(1)
        out = self.relu(self.bn(self.fc(out)))
        out = self.drop(out)
        out = self.linear(out)
        return out
    
# For default setting, create and summarize an SPGNet model.
if __name__ == "__main__":
    model = SPGNet(num_classes=12)
    # Print a summary of the model architecture.
    paddle.summary(model, (1, 3, 32, 32))