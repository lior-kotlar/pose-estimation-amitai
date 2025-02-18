import json
from torchsummary import summary
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F


class Encoder2DAtrous(nn.Module):
    def __init__(self, img_size, filters, kernel_size, dilation_rate, dropout):
        super(Encoder2DAtrous, self).__init__()
        self.image_size = img_size
        self.dilation_rate = int(dilation_rate)
        self.dropout = int(dropout)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.output_ratio = 4
        self.padding = 2

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(float(self.dropout))

        self.conv1 = self.get_conv2d(input_channels=img_size[-1], num_filters= self.filters)
        self.bn1 = nn.BatchNorm2d(self.filters)
        self.conv2 = self.get_conv2d(input_channels=self.filters, num_filters= self.filters)
        self.bn2 = nn.BatchNorm2d(self.filters)
        self.conv3 = self.get_conv2d(input_channels=self.filters, num_filters= self.filters)
        self.bn3 = nn.BatchNorm2d(self.filters)

        self.conv4 = self.get_conv2d(input_channels=self.filters, num_filters=self.filters * 2)
        self.bn4 = nn.BatchNorm2d(self.filters * 2)
        self.conv5 = self.get_conv2d(input_channels=self.filters * 2, num_filters=self.filters * 2)
        self.bn5 = nn.BatchNorm2d(self.filters * 2)
        self.conv6 = self.get_conv2d(input_channels=self.filters * 2, num_filters=self.filters * 2)
        self.bn6 = nn.BatchNorm2d(self.filters * 2)

        self.conv7 = self.get_conv2d(input_channels=self.filters * 2, num_filters=self.filters * 4)
        self.bn7 = nn.BatchNorm2d(self.filters * 4)
        self.conv8 = self.get_conv2d(input_channels=self.filters * 4, num_filters=self.filters * 4)
        self.bn8 = nn.BatchNorm2d(self.filters * 4)
        self.conv9 = self.get_conv2d(input_channels=self.filters * 4, num_filters=self.filters * 4)
        self.bn9 = nn.BatchNorm2d(self.filters * 4)

    def get_conv2d(self, num_filters, input_channels):
        conv = nn.Conv2d(in_channels=int(input_channels), out_channels=int(num_filters),
                         kernel_size=int(self.kernel_size), padding=int(self.padding), dilation=int(self.dilation_rate))
        # nn.init.xavier_normal_(conv.weight)
        return conv

    def get_output_size(self):
        return (self.image_size[0] // self.output_ratio,
                self.image_size[1] // self.output_ratio,
                self.filters * self.output_ratio)

    # def forward(self, x):
    #     x1 = self.leakyrelu(self.bn1(self.conv1(x)))
    #     x2 = self.leakyrelu(self.bn2(self.conv2(x1))) + x1  # Skip connection between conv1 and conv2
    #     x3 = self.leakyrelu(self.bn3(self.conv3(x2))) + x2  # Skip connection between conv2 and conv3
    #     x = self.dropout(self.leakyrelu(self.maxpool(x3)))
    #
    #     x4 = self.leakyrelu(self.bn4(self.conv4(x)))
    #     x5 = self.leakyrelu(self.bn5(self.conv5(x4))) + x4  # Skip connection between conv4 and conv5
    #     x6 = self.leakyrelu(self.bn6(self.conv6(x5))) + x5  # Skip connection between conv5 and conv6
    #     x = self.dropout(self.leakyrelu(self.maxpool(x6)))
    #
    #     x7 = self.leakyrelu(self.bn7(self.conv7(x)))
    #     x8 = self.leakyrelu(self.bn8(self.conv8(x7))) + x7  # Skip connection between conv7 and conv8
    #     x9 = self.leakyrelu(self.bn9(self.conv9(x8))) + x8  # Skip connection between conv8 and conv9
    #     x = self.dropout(x9)
    #     return x

    def forward(self, x):
        x1 = self.leakyrelu(self.conv1(x))
        x2 = self.leakyrelu(self.conv2(x1)) + x1  # Skip connection between conv1 and conv2
        x3 = self.leakyrelu(self.conv3(x2)) + x2  # Skip connection between conv2 and conv3
        x = self.dropout(self.leakyrelu(self.maxpool(x3)))

        x4 = self.leakyrelu(self.conv4(x))
        x5 = self.leakyrelu(self.conv5(x4)) + x4  # Skip connection between conv4 and conv5
        x6 = self.leakyrelu(self.conv6(x5)) + x5  # Skip connection between conv5 and conv6
        x = self.dropout(self.leakyrelu(self.maxpool(x6)))

        x7 = self.leakyrelu(self.conv7(x))
        x8 = self.leakyrelu(self.conv8(x7)) + x7  # Skip connection between conv7 and conv8
        x9 = self.leakyrelu(self.conv9(x8)) + x8  # Skip connection between conv8 and conv9
        x = self.dropout(x9)
        return x



class Decoder2d(nn.Module):
    def __init__(self, input_shape, num_output_channels,
                 kernel_size,
                 filters,
                 dropout):
        super(Decoder2d, self).__init__()
        self.input_shape = input_shape
        self.num_output_channels = int(num_output_channels)
        self.dropout = float(dropout)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(self.dropout)

        self.conv2dTranspose1 = nn.ConvTranspose2d(in_channels=self.input_shape[-1], out_channels=input_shape[-1] // 2,
                        kernel_size=self.kernel_size,
                        stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(input_shape[-1] // 2)

        self.conv2dTranspose2 = nn.ConvTranspose2d(in_channels=self.input_shape[-1] // 2,
                                                   out_channels=input_shape[-1] // 2,
                        kernel_size=self.kernel_size,
                        stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(input_shape[-1] // 2)

        self.conv2dTranspose3 = nn.ConvTranspose2d(in_channels=self.input_shape[-1] // 2,
                                                   out_channels=input_shape[-1] // 2,
                        kernel_size=self.kernel_size,
                        stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(input_shape[-1] // 2)

        self.conv2dTranspose4 = nn.ConvTranspose2d(in_channels=self.input_shape[-1] // 2,
                                                   out_channels=num_output_channels,
                        kernel_size=self.kernel_size,
                        stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(num_output_channels)

    @staticmethod
    def normalize_between_0_and_1(x):
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def get_conv2d_transpose(self, in_channels, out_channels, stride):
        conv = nn.ConvTranspose2d(in_channels=int(in_channels), out_channels=int(out_channels),
                                  kernel_size=int(self.kernel_size),
                                  stride=int(stride), padding=1, output_padding=1)
        # nn.init.xavier_normal_(conv.weight)
        return conv

    # def forward(self, x):
    #     x1 = self.leakyrelu(self.bn1(self.conv2dTranspose1(x)))
    #     x2 = self.leakyrelu(self.bn2(self.conv2dTranspose2(x1))) + x1
    #     x3 = self.leakyrelu(self.bn3(self.conv2dTranspose3(x2))) + x2
    #     x = self.leakyrelu(self.bn4(self.conv2dTranspose4(x3)))
    #     # x = self.normalize_between_0_and_1(x)
    #     return x

    def forward(self, x):
        x1 = self.leakyrelu(self.conv2dTranspose1(x))
        x2 = self.leakyrelu(self.conv2dTranspose2(x1)) + x1
        x3 = self.leakyrelu(self.conv2dTranspose3(x2)) + x2
        x = self.leakyrelu(self.conv2dTranspose4(x3))
        # x = self.normalize_between_0_and_1(x)
        return x


class BasicNet(nn.Module):
    def __init__(self, config, image_size, number_of_output_channels):
        super(BasicNet, self).__init__()
        self.config = config
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.kernel_size = config["convolution kernel size"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]

        self.encoder = Encoder2DAtrous(img_size=self.image_size,
                                       filters=self.num_base_filters,
                                       kernel_size=self.kernel_size,
                                       dilation_rate=self.dilation_rate,
                                       dropout=self.dropout)
        self.decoder = Decoder2d(input_shape=self.encoder.get_output_size(),
                                 filters=self.num_base_filters,
                                 kernel_size=self.kernel_size,
                                 dropout=self.dropout,
                                 num_output_channels=self.number_of_output_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FourCamerasBaseLine(nn.Module):
    def __init__(self, config, image_size, number_of_output_channels):
        super(FourCamerasBaseLine, self).__init__()
        self.config = config
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.kernel_size = config["convolution kernel size"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]

        self.shared_encoder = Encoder2DAtrous(img_size=(image_size[0], image_size[1], image_size[2] // 4),
                                       filters=self.num_base_filters,
                                       kernel_size=self.kernel_size,
                                       dilation_rate=self.dilation_rate,
                                       dropout=self.dropout)

        self.shared_conv2d = nn.Conv2d(self.shared_encoder.get_output_size()[-1] * 4,
                                       self.shared_encoder.get_output_size()[-1] * 4,
                                  kernel_size=1,
                                  padding=0, bias=True)

        input_size = list(self.shared_encoder.get_output_size())
        input_size[-1] *= 5
        self.shared_decoder = Decoder2d(input_shape=input_size,
                                   num_output_channels=self.number_of_output_channels // 4,
                                   kernel_size=self.kernel_size,
                                   filters=self.num_base_filters,
                                   dropout=self.dropout)

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(x, 4, dim=1)

        enc1 = self.shared_encoder(x1)
        enc2 = self.shared_encoder(x2)
        enc3 = self.shared_encoder(x3)
        enc4 = self.shared_encoder(x4)

        all_encoders = torch.cat((enc1, enc2, enc3, enc4), dim=1)
        all_encoders = self.shared_conv2d(all_encoders) + all_encoders

        decoded1 = self.shared_decoder(torch.cat((enc1, all_encoders), dim=1))
        decoded2 = self.shared_decoder(torch.cat((enc2, all_encoders), dim=1))
        decoded3 = self.shared_decoder(torch.cat((enc3, all_encoders), dim=1))
        decoded4 = self.shared_decoder(torch.cat((enc4, all_encoders), dim=1))

        output = torch.cat((decoded1, decoded2, decoded3, decoded4), dim=1)
        return output


class FourCamerasDisentanglement(nn.Module):
    def __init__(self, config, image_size, number_of_output_channels):
        super(FourCamerasDisentanglement, self).__init__()
        self.config = config
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.kernel_size = config["convolution kernel size"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]

        self.shared_encoder = Encoder2DAtrous(img_size=(image_size[0], image_size[1], image_size[2] // 4),
                                              filters=self.num_base_filters,
                                              kernel_size=self.kernel_size,
                                              dilation_rate=self.dilation_rate,
                                              dropout=self.dropout)

        self.rearrange_layer_1 = nn.Conv2d(in_channels=int(self.shared_encoder.get_output_size()[-1]),
                                   out_channels=300,
                                   kernel_size=1,
                                   padding=0)
        self.FTL = FTL()

        self.fusion_layer_1 = nn.Conv2d(in_channels=1600, out_channels=400, kernel_size=1, padding=0)
        self.fusion_layer_2 = nn.Conv2d(in_channels=400, out_channels=400, kernel_size=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(400)
        self.batch_norm2 = nn.BatchNorm2d(400)
        self.batch_norm3 = nn.BatchNorm2d(300)
        self.relu = nn.ReLU(inplace=True)

        self.invFTL = InvFTL()

        self.rearrange_layer_2 = nn.Conv2d(in_channels=300,
                                   out_channels=int(self.shared_encoder.get_output_size()[-1]),
                                   kernel_size=1,
                                   padding=0)

        self.shared_decoder = Decoder2d(input_shape=self.shared_encoder.get_output_size(),
                                        num_output_channels=int(self.number_of_output_channels // 4),
                                        kernel_size=int(self.kernel_size),
                                        filters=int(self.num_base_filters),
                                        dropout=float(self.dropout))

    def forward(self, x, camera_matrices, camera_matrices_inv):
        x1, x2, x3, x4 = torch.split(x, 4, dim=1)

        first_encoder_1 = self.shared_encoder(x1)
        first_encoder_2 = self.shared_encoder(x2)
        first_encoder_3 = self.shared_encoder(x3)
        first_encoder_4 = self.shared_encoder(x4)

        enc1 = self.rearrange_layer_1(first_encoder_1)
        enc2 = self.rearrange_layer_1(first_encoder_2)
        enc3 = self.rearrange_layer_1(first_encoder_3)
        enc4 = self.rearrange_layer_1(first_encoder_4)

        canonic_1 = self.invFTL(enc1, camera_matrices_inv[:, 0])
        canonic_2 = self.invFTL(enc2, camera_matrices_inv[:, 1])
        canonic_3 = self.invFTL(enc3, camera_matrices_inv[:, 2])
        canonic_4 = self.invFTL(enc4, camera_matrices_inv[:, 3])

        can_fusion = torch.cat([canonic_1, canonic_2, canonic_3, canonic_4], dim=1)
        can_fusion = self.relu(self.batch_norm1(self.fusion_layer_1(can_fusion)))
        can_fusion = self.relu(self.batch_norm2(self.fusion_layer_2(can_fusion)))

        entangled_v1 = self.relu(self.batch_norm3(self.FTL(can_fusion, camera_matrices[:, 0])))
        entangled_v2 = self.relu(self.batch_norm3(self.FTL(can_fusion, camera_matrices[:, 1])))
        entangled_v3 = self.relu(self.batch_norm3(self.FTL(can_fusion, camera_matrices[:, 2])))
        entangled_v4 = self.relu(self.batch_norm3(self.FTL(can_fusion, camera_matrices[:, 3])))

        entangled_v1 = self.rearrange_layer_2(entangled_v1)
        entangled_v2 = self.rearrange_layer_2(entangled_v2)
        entangled_v3 = self.rearrange_layer_2(entangled_v3)
        entangled_v4 = self.rearrange_layer_2(entangled_v4)

        # do also skip connection
        out_1 = self.shared_decoder(entangled_v1 + first_encoder_1)
        out_2 = self.shared_decoder(entangled_v2 + first_encoder_2)
        out_3 = self.shared_decoder(entangled_v3 + first_encoder_3)
        out_4 = self.shared_decoder(entangled_v4 + first_encoder_4)

        output = torch.cat((out_1, out_2, out_3, out_4), dim=1)

        return output




class FTL(nn.Module):
    def __init__(self):
        super(FTL, self).__init__()
        pass

    def forward(self, x, P):
        z = torch.reshape(x, (-1, 48, 48, 100, 4, 1))
        P = torch.reshape(P, (-1, 1, 1, 1, 3, 4))
        projected = P @ z
        projected = torch.reshape(projected, (-1, 300, 48, 48))
        return projected



class InvFTL(nn.Module):
    def __init__(self):
        super(InvFTL, self).__init__()

    def forward(self, x, inv_P):
        z = torch.reshape(x, (-1, 48, 48, 100, 3, 1))
        P = torch.reshape(inv_P, (-1, 1, 1, 1, 4, 3))
        inverted = P @ z
        inverted = torch.reshape(inverted, (-1, 400, 48, 48))
        return inverted




if __name__ == "__main__":
    image_size = (192, 192, 16)
    filters = 64
    num_blocks = 2
    kernel_size = 3
    dilation_rate = 2
    dropout = 0.5
    # encoder = Encoder2DAtrous(img_size=image_size,filters=filters,
    #                           kernel_size=kernel_size,
    #                           dilation_rate=dilation_rate,dropout=dropout)
    #
    # decoder = Decoder2d(input_shape=encoder.get_output_size(), filters=filters,
    #                     kernel_size=kernel_size, dropout=dropout, num_output_channels=10)
    configuration_path = '../tensorflow/train_config.json'
    with open(configuration_path) as C:
        config = json.load(C)
    x0 = torch.randn(10, 16, 192, 192)
    four_cams = FourCamerasBaseLine(config=config,
                         image_size=image_size,
                         number_of_output_channels=40)
    trainable_params = sum(p.numel() for p in four_cams.parameters() if p.requires_grad)
    summary(four_cams, x0.shape[1:])
    pass


    Ps = torch.rand(10, 4, 3, 4)
    inv_Ps = torch.rand(10, 4, 4, 3)
    y0 = four_cams(x0, Ps, inv_Ps)
    pass


