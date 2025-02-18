import numpy as np
from scipy.ndimage import shift, rotate
from PIL import Image
# import tensorflow as tf
import torch
from constants import *
import cv2


class Augmentor:
    @staticmethod
    def scale_image(img, scale):
        if scale != 1:
            height, width = img.shape[:2]  # Get the height and width of the image
            center = (width / 2, height / 2)  # Get the center of the image

            # Generate a rotation matrix with no rotation, only scaling
            rotation_matrix = cv2.getRotationMatrix2D(center, angle=0, scale=scale)

            # Apply the transformation (rotation and scaling)
            zoomed_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            zoomed_img = img
        # plt.figure()
        # plt.imshow(zoomed_img)
        # plt.axis('equal')
        # plt.show()
        return zoomed_img

    @staticmethod
    def augment(img, h_fl, v_fl, rotation_angle, shift_y_x, scale_factor):
        if h_fl:
            img = np.fliplr(np.copy(img))
        if v_fl:
            img = np.flipud(np.copy(img))
        if scale_factor != 1:
            img = Augmentor.scale_image(np.copy(img), scale_factor)
        if not np.all(shift_y_x == 0):
            img = shift(np.copy(img), shift_y_x)
        if rotation_angle != 0:
            img = Augmentor.rotate_scipy(np.copy(img), rotation_angle)
        return img

    @staticmethod
    def shift_image(img, shift_y_x):
        shift_x, shift_y = shift_y_x
        height, width = img.shape

        # Find non-zero pixel indices
        nz_y, nz_x = np.nonzero(img)

        # If there are no non-zero pixels, return the original image
        if len(nz_y) == 0 or len(nz_x) == 0:
            return img

        # Calculate boundaries of the non-zero pixels
        min_y, max_y = np.min(nz_y), np.max(nz_y)
        min_x, max_x = np.min(nz_x), np.max(nz_x)

        # Calculate the maximum allowable shift
        max_shift_y_down = height - 1 - max_y
        max_shift_y_up = -min_y
        max_shift_x_right = width - 1 - max_x
        max_shift_x_left = -min_x

        # Adjust shift_y and shift_x to stay within bounds
        shift_y = min(max(shift_y, max_shift_y_up), max_shift_y_down)
        shift_x = min(max(shift_x, max_shift_x_left), max_shift_x_right)

        shifted_image = np.zeros_like(img)

        # Calculate new indices after the shift
        new_y = np.clip(nz_y + shift_y, 0, height - 1)
        new_x = np.clip(nz_x + shift_x, 0, width - 1)

        # Place the non-zero values into the new positions
        shifted_image[new_y, new_x] = img[nz_y, nz_x]
        # from matplotlib import pyplot as plt
        # plt.imshow(img*0.5 + shifted_image)
        # plt.show()
        return shifted_image

    @staticmethod
    def rotate(img, rotation_angle):
        img_pil = Image.fromarray(img)
        img_pil = img_pil.rotate(rotation_angle, 3)
        img = np.asarray(img_pil)
        return img

    @staticmethod
    def rotate_scipy(img, rotation_angle):
        rotated_image = rotate(img, rotation_angle, reshape=False, order=3)
        return rotated_image

    @staticmethod
    def custom_augmentations(img, rotation_angle, shift_y_x, do_horizontal_flip, do_vertical_flip, scaling):
        num_channels = img.shape[-1]
        augmented_image = np.zeros_like(img)
        for channel in range(num_channels):
            augmented_image[:, :, channel] = Augmentor.augment(np.copy(img[:, :, channel]), do_horizontal_flip,
                                                   do_vertical_flip, rotation_angle, shift_y_x, scaling)
        return augmented_image

    @staticmethod
    def tf_find_peaks(x):
        """
        Finds the maximum value in each channel and returns the location and value.
        Args:
            x: rank-4 tensor (samples, height, width, channels)

        Returns:
            peaks: rank-3 tensor (samples, [x, y], channels)
        """
        # Store input shape
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)

            # Ensure x is on the correct device
        x = x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Store input shape
        in_shape = x.shape

        image_size = int(in_shape[1])

        # Flatten height/width dims
        flattened = x.reshape(in_shape[0], in_shape[1] * in_shape[2], in_shape[3])

        # Find peaks in linear indices
        vals, idx = torch.max(flattened, dim=1)

        # Convert linear indices to subscripts
        rows = idx // in_shape[2]
        cols = idx % in_shape[2]

        # Return N x 3 x C tensor
        pred = torch.stack([
            cols.float(),
            rows.float(),
            vals
        ], dim=1)

        pred = pred.permute(0, 2, 1)
        pred = pred[..., :2]
        # pred = pred / image_size  # normalize points

        return pred

