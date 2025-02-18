import numpy as np
from scipy.ndimage import shift
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


class Augmentor:
    def __init__(self,
                 config, number_of_input_channels, number_of_output_channels):
        self.use_custom_function = bool(config["custom"])
        self.xy_shifts = config["augmentation shift x y"]
        self.rotation_range = config["rotation range"]
        self.seed = config["seed"]
        self.batch_size = config["batch_size"]
        self.zoom_range = config["zoom range"]
        self.interpolation_order = config["interpolation order"]
        self.do_horizontal_flip = bool(config["horizontal flip"])
        self.do_vertical_flip = bool(config["vertical flip"])
        self.number_of_input_channels = number_of_input_channels
        self.number_of_output_channels = number_of_output_channels
        self.custom_augmentation_function = self.get_custom_augmentation_function()
        self.debug_mode = bool(config["debug mode"])

        if self.debug_mode:
            self.batch_size = 1

    def get_data_generator(self, box, confmaps):
        datagen = self.config_data_generator(box, confmaps)
        return datagen

    def config_data_generator(self, box, confmaps):
        if self.use_custom_function:
            data_gen_args = dict(preprocessing_function=self.custom_augmentation_function)
        else:
            data_gen_args = dict(rotation_range=self.rotation_range,
                                 zoom_range=self.zoom_range,
                                 horizontal_flip=self.do_horizontal_flip,
                                 vertical_flip=self.do_vertical_flip,
                                 width_shift_range=self.xy_shifts,
                                 height_shift_range=self.xy_shifts,
                                 interpolation_order=self.interpolation_order)

        datagen_x = ImageDataGenerator(**data_gen_args)
        datagen_y = ImageDataGenerator(**data_gen_args)

        datagen_x.fit(box, augment=True, seed=self.seed)
        datagen_y.fit(confmaps, augment=True, seed=self.seed)

        flow_box = datagen_x.flow(box, batch_size=self.batch_size, seed=self.seed, shuffle=False)
        flow_conf = datagen_y.flow(confmaps, batch_size=self.batch_size, seed=self.seed, shuffle=False)

        return self.custom_generator(flow_box, flow_conf)

    def custom_generator(self, flow_box, flow_conf):
        while True:
            box_batch = next(flow_box)
            conf_batch = next(flow_conf)
            yield box_batch, conf_batch

    @staticmethod
    def augment(img, h_fl, v_fl, rotation_angle, shift_y_x, scale_factor):
        if np.max(img) <= 1:
            img = np.uint8(img * 255)
        if h_fl:
            img = np.fliplr(img)
        if v_fl:
            img = np.flipud(img)
        if scale_factor != 1:
            img = Augmentor.scale(img, scale_factor=scale_factor)
        img = shift(img, shift_y_x)
        img_pil = Image.fromarray(img)
        img_pil = img_pil.rotate(rotation_angle, 3)
        img = np.asarray(img_pil)
        if np.max(img) > 1:
            img = img / 255
        return img

    @staticmethod
    def scale(img, scale_factor):
        N = img.shape[0]
        scale_factor = float(scale_factor)
        center = (N / 2, N / 2)
        zoom_matrix = cv2.getRotationMatrix2D(center, 0, scale_factor)
        zoomed_img = cv2.warpAffine(img, zoom_matrix, (N, N), flags=cv2.INTER_CUBIC)
        return zoomed_img

    def get_custom_augmentation_function(self):
        rotation_range = self.rotation_range
        xy_shift = self.xy_shifts
        can_horizontal_flip = self.do_horizontal_flip
        can_vertical_flip = self.do_vertical_flip
        zoom_range = self.zoom_range
        number_of_input_channels = self.number_of_input_channels
        number_of_output_channels = self.number_of_output_channels

        def custom_augmentations(img):
            new_image = np.zeros_like(img)
            do_horizontal_flip = bool(np.random.randint(2)) and can_horizontal_flip
            do_vertical_flip = bool(np.random.randint(2)) and can_vertical_flip
            rotation_angle = np.random.randint(-rotation_range, rotation_range)
            shift_y_x = np.random.randint(-xy_shift, xy_shift, 2)
            random_scale = np.random.uniform(low=zoom_range[0], high=zoom_range[1], size=1)
            num_channels = img.shape[-1]
            for channel in range(num_channels):
                new_image[:, :, channel] = Augmentor.augment(np.copy(img[:, :, channel]), do_horizontal_flip,
                                                             do_vertical_flip, rotation_angle, shift_y_x, random_scale)
            return new_image

        return custom_augmentations
