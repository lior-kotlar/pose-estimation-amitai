import json
from torchsummary import summary
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F

class PretrainedResnetEncoder(nn.Module):
    def __init__(self, config, image_size, number_of_output_channels):
        super(PretrainedResnetEncoder, self).__init__()
        self.config = config
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.kernel_size = config["convolution kernel size"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]

        model = models.resnet50(pretrained=True)
        # Get the children of the model
        children = list(model.children())
        modules = children[:-4]

        self.encoder = torch.nn.Sequential(*modules)
        self.decoder = Decoder2d(input_shape=(12, 12, 512),
                                 filters=self.num_base_filters,
                                 kernel_size=self.kernel_size,
                                 dropout=self.dropout,
                                 num_output_channels=self.number_of_output_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PretrainedLEAP(nn.Module):
    def __init__(self, config, image_size, number_of_output_channels):
        super(PretrainedLEAP, self).__init__()
        self.config = config
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.model = BasicNet(self.config, self.image_size, 4)
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.kernel_size = config["convolution kernel size"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]
        path = r"pretrained models/pre_trained_leap.pth"
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.encoder = self.model.encoder
        self.decoder = Decoder2d(input_shape=self.encoder.get_output_size(),
                                 num_output_channels=self.number_of_output_channels,
                                 kernel_size=self.kernel_size,
                                 filters=self.num_base_filters,
                                 dropout=self.dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


class GPTResNetEncoderDecoder(nn.Module):
    def __init__(self,image_size=(192, 192, 4), num_output_channels=10):
        super(GPTResNetEncoderDecoder, self).__init__()

        # Define the initial convolutional layer to handle 4-channel input
        self.initial = nn.Conv2d(image_size[-1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder - ResNet Blocks
        self.encoder_block1 = self.make_layer(64, 64, stride=1, num_blocks=2)
        self.encoder_block2 = self.make_layer(64, 128, stride=2, num_blocks=2)
        self.encoder_block3 = self.make_layer(128, 256, stride=2, num_blocks=2)
        self.encoder_block4 = self.make_layer(256, 512, stride=2, num_blocks=2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_block4 = self.make_layer(256, 256, stride=1, num_blocks=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_block3 = self.make_layer(128, 128, stride=1, num_blocks=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_block2 = self.make_layer(64, 64, stride=1, num_blocks=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder_block1 = self.make_layer(64, 64, stride=1, num_blocks=2)

        # Final deconvolution
        self.final_deconv = nn.ConvTranspose2d(64, num_output_channels, kernel_size=1)

    def make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.initial_pool(self.initial_relu(self.initial_bn(self.initial(x))))

        # Encoder
        skip_conn1 = x = self.encoder_block1(x)
        skip_conn2 = x = self.encoder_block2(x)
        skip_conn3 = x = self.encoder_block3(x)
        x = self.encoder_block4(x)

        # Decoder
        x = self.upconv4(x)
        x = self.decoder_block4(x + skip_conn3)
        x = self.upconv3(x)
        x = self.decoder_block3(x + skip_conn2)
        x = self.upconv2(x)
        x = self.decoder_block2(x + skip_conn1)
        x = self.upconv1(x)

        # Final deconv to get the right number of output channels
        x = self.final_deconv(x)

        # Interpolate to the desired output size
        x = F.interpolate(x, size=(192, 192), mode='bicubic', align_corners=False)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
