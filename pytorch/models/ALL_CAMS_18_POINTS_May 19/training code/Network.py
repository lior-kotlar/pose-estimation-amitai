from constants import *
import VITs
import CNNs
from torchsummary import summary
import torch

class Network:
    def __init__(self, config, image_size, num_output_channels):
        self.config = config
        self.image_size = np.array(image_size)
        self.num_output_channels = num_output_channels
        self.model_type = self.config['model type']
        self.model = self.config_model()

    def config_model(self):
        if (self.model_type == MODEL_18_POINTS_PER_WING or self.model_type == MODEL_18_POINTS_3_GOOD_CAMERAS or
                self.model_type == ALL_POINTS_MODEL):
            return CNNs.BasicNet(self.config, self.image_size, self.num_output_channels)
        elif self.model_type == MODEL_18_POINTS_PER_WING_VIT:
            return VITs.VIT_encoder_CNN_decoder(self.config, self.image_size, self.num_output_channels)
        elif self.model_type == ALL_CAMS_18_POINTS:
            return CNNs.FourCamerasBaseLine(self.config, self.image_size, self.num_output_channels)
        elif self.model_type == ALL_CAMS_DISENTANGLED_PER_WING_CNN:
            return CNNs.FourCamerasDisentanglement(self.config, self.image_size, self.num_output_channels)
        elif self.model_type == ALL_CAMS_18_POINTS_VIT:
            return VITs.VIT4CamerasBaseLine(self.config, self.image_size, self.num_output_channels)

    def get_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if self.model_type == ALL_CAMS_DISENTANGLED_PER_WING_CNN:
            summary(model=self.model, input_size=[(self.image_size[-1], self.image_size[0], self.image_size[1]),
                                                  (4, 3, 4), (4, 4, 3)])
        else:
            summary(model=self.model, input_size=(self.image_size[-1], self.image_size[0], self.image_size[1]))
        return self.model
