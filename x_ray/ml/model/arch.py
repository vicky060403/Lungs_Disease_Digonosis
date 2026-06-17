# Architecture of Custom CNN for lungs images
import torch
import torch.nn as nn

# custom cnn
class Net(nn.Module):
    """
    Custom CNN Arcitecture for Image Classification

    """
    # Inherit and initialize the parent nn.Module class
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3,3),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.pooling1 = nn.MaxPool2d(2, 2)
        # Conv block 1
        self.conv_block11 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(20)
        )
        self.pooling11 = nn.MaxPool2d(2, 2)

        self.conv_block12 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )
        self.pooling12 = nn.MaxPool2d(2, 2)
        # Conv block 2
        self.conv_block21 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(20)
        )

        self.conv_block22 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.conv_block23 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )

        self.conv_block24 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )

        self.conv_block25 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=14,
                kernel_size=(3, 3),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )

        self.conv_block26 = nn.Sequential(
            nn.Conv2d(
                in_channels=14,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1)
        # classification layer
        self.classifier = nn.Conv2d(
            in_channels=16,
            out_channels=2,
            kernel_size=(1, 1)
        )

    # forward pass
    def forward(self, x):
        # input block
        x = self.conv_block1(x)
        x = self.pooling1(x)
        # conv block1
        x = self.conv_block11(x)
        x = self.pooling11(x)
        x = self.conv_block12(x)
        x = self.pooling12(x)
        # conv block2
        x = self.conv_block21(x)
        x = self.conv_block22(x)
        x = self.conv_block23(x)
        x = self.conv_block24(x)
        x = self.conv_block25(x)
        x = self.conv_block26(x)
        # average polling
        x = self.gap(x)
        # output layer having 2 neurons(2 class classification layer)
        x = self.classifier(x)
        # flatten the layer
        x = torch.flatten(x, 1)

        return x        
