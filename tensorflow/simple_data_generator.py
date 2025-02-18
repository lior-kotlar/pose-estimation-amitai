
import numpy as np
from Augmentor import Augmentor
import h5py
from scipy.ndimage import binary_dilation
from matplotlib import pyplot as plt

class SimpleDataGenerator:
    def __init__(self,
                 config,
                 box,
                 confmaps,
                 validation_phase=False,):
        self.do_augmentations = bool(config["do augmentations"])
        self.use_custom_function = bool(config["custom"])
        self.xy_shifts = config["augmentation shift x y"]
        self.rotation_range = config["rotation range"]
        self.seed = config["seed"]
        self.batch_size = config["batch_size"]
        self.zoom_range = config["zoom range"]
        self.model_type = config["model type"]
        self.interpolation_order = config["interpolation order"]
        self.do_horizontal_flip = bool(config["horizontal flip"])
        self.do_vertical_flip = bool(config["vertical flip"])
        self.wings_masks_dilation = config["wings_masks_dilation"]
        self.validation_phase = validation_phase
        np.random.seed(self.seed)
        self.box = box
        self.confmaps = confmaps

    def generate(self):
        is_new_epoch = True
        while True:
            if is_new_epoch:
                # Shuffle indices at the start of each epoch
                indices = np.arange(len(self.box))
                if not self.validation_phase:
                   np.random.shuffle(indices)
                is_new_epoch = False  # Reset the flag

            for i in range(0, len(indices), self.batch_size):
                # Get the indices for this batch
                batch_indices = indices[i:i + self.batch_size]
                if self.validation_phase:
                    print("Batch index: ", batch_indices, flush=True)

                # Check if we're at the end of the epoch
                if i + self.batch_size >= len(indices):
                    is_new_epoch = True  # Set the flag for the next iteration

                # If the batch size is not divisible by the number of samples,
                # take samples from the beginning to make up a full batch
                if len(batch_indices) < self.batch_size:
                    batch_indices = np.concatenate([batch_indices, indices[:(self.batch_size - len(batch_indices))]])

                batch_images = []
                batch_confmaps = []

                for idx in batch_indices:
                    image = self.box[idx]
                    confmap = self.confmaps[idx]

                    # Apply your augmentation function
                    if not self.validation_phase and self.do_augmentations:
                        confmap, image = self.perform_augmentations(confmap.copy(), image.copy())

                    batch_images.append(image)
                    batch_confmaps.append(confmap)

                yield np.array(batch_images), np.array(batch_confmaps)

    def perform_augmentations(self, confmap, image):
        do_horizontal_flip = bool(np.random.randint(2)) and self.do_horizontal_flip
        do_vertical_flip = bool(np.random.randint(2)) and self.do_vertical_flip
        if self.rotation_range > 0:
            rotation_angle = np.random.randint(-self.rotation_range, self.rotation_range)
        else:
            rotation_angle = 0
        if self.xy_shifts > 0:
            shift_y_x = np.random.randint(-self.xy_shifts, self.xy_shifts, 2)
        else:
            shift_y_x = 0
        scaling = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        if np.random.randint(2) and bool(self.wings_masks_dilation):
            mask_dilation_size = np.random.randint(0, self.wings_masks_dilation)
        else:
            mask_dilation_size = 0
        augmented_image = self.augment_image(image, do_horizontal_flip, do_vertical_flip, rotation_angle, shift_y_x,
                                             scaling, mask_dilation_size)
        augmented_confmap = self.augment_image(confmap, do_horizontal_flip, do_vertical_flip, rotation_angle,
                                               shift_y_x,
                                               scaling,
                                               mask_dilation_size=0)
        # augmented_confmap = SimpleDataGenerator.ensure_sigma(augmented_confmap)
        return augmented_confmap, augmented_image

    @staticmethod
    def augment_image(img, do_horizontal_flip, do_vertical_flip, rotation_angle, shift_y_x, scaling, mask_dilation_size=0):
        augmented_image = np.zeros_like(img)
        num_channels = img.shape[-1]
        for channel in range(num_channels):
            augmented_image[:, :, channel] = Augmentor.augment(img[:, :, channel].copy(), do_horizontal_flip,
                                                   do_vertical_flip, rotation_angle, shift_y_x, scaling)
        if bool(mask_dilation_size):
            if img.shape[-1] == 4:
                masks_inds = [-1]
            elif img.shape[-1] == 5:
                masks_inds = [-2, -1]
            elif img.shape[-1] == 16:
                masks_inds = [3, 7, 11, 15]
            else:
                assert "shape must be 4 or 5 or 16"
            for mask_ind in masks_inds:
                mask = img[:, :, mask_ind].copy()
                mask = binary_dilation(mask, iterations=mask_dilation_size)
                augmented_image[:, :, mask_ind] = mask
        return augmented_image

    @staticmethod
    def get_gaussian(mean, sigma=3, grid_size=(192, 192)):
        x, y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]))
        d = np.sqrt((x - mean[0]) ** 2 + (y - mean[1]) ** 2)
        # Gaussian function
        g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
        return g

    @staticmethod
    def ensure_sigma(confmaps, sigma=3):
        points_2d = Augmentor.tf_find_peaks(confmaps[np.newaxis, ...]).squeeze()
        new_confmaps = np.zeros_like(confmaps)
        num_channels = confmaps.shape[-1]
        for channel in range(num_channels):
            mean = points_2d[channel]
            new_confmap = SimpleDataGenerator.get_gaussian(mean, sigma=3)
            new_confmaps[..., channel] = new_confmap
        return new_confmaps

