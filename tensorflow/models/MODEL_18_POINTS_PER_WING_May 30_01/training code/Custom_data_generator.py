import matplotlib.pyplot as plt
from scipy.ndimage import shift
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from constants import *
import cv2
import numpy as np
from Augmentor import Augmentor


class CustomDataGenerator:
    def __init__(self,
                 config,
                 box,
                 confmaps,
                 validation_phase=False,
                 points_3D=None):
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
        self.validation_phase = validation_phase
        self.box = box
        self.points_3D = points_3D
        self.confmaps = confmaps

    def generate(self):
        if (self.model_type == ALL_CAMS_DISENTANGLED_PER_WING_CNN or
                self.model_type == ALL_CAMS_DISENTANGLED_PER_WING_VIT):
            yield from self.generate_with_camera_matrices()
        else:
            yield from self.generate_box_confmaps()

    # def generate_box_confmaps(self):
    #     while True:
    #         if self.validation_phase:
    #             batch_images = self.box
    #             batch_confmaps = self.confmaps
    #             yield np.array(batch_images), np.array(batch_confmaps)
    #
    #         # Shuffle indices at the start of each epoch
    #         indices = np.arange(len(self.box))
    #         np.random.shuffle(indices)
    #
    #         for i in range(0, len(indices), self.batch_size):
    #             # Get the indices for this batch
    #             batch_indices = indices[i:i + self.batch_size]
    #
    #             # If the batch size is not divisible by the number of samples,
    #             # take samples from the beginning to make up a full batch
    #             if len(batch_indices) < self.batch_size:
    #                 batch_indices = np.concatenate([batch_indices, indices[:(self.batch_size - len(batch_indices))]])
    #
    #             batch_images = []
    #             batch_confmaps = []
    #
    #             for idx in batch_indices:
    #                 image = self.box[idx]
    #                 confmap = self.confmaps[idx]
    #
    #                 # Apply your augmentation function
    #                 if not self.validation_phase:
    #                     confmap, image = self.do_augmentations(confmap, image)
    #
    #                 # import matplotlib
    #                 # matplotlib.use('TkAgg')
    #                 # plt.imshow(image[..., 1] + np.sum(confmap, axis=-1))
    #                 # plt.show()
    #
    #                 batch_images.append(image)
    #                 batch_confmaps.append(confmap)
    #
    #             output = (np.array(batch_images),
    #                       np.array(batch_confmaps))
    #             yield output

    def generate_box_confmaps(self):
        # Initialize the epoch flag
        is_new_epoch = True

        while True:
            if is_new_epoch:
                # Shuffle indices at the start of each epoch
                indices = np.arange(len(self.box))
                np.random.shuffle(indices)
                is_new_epoch = False  # Reset the flag

            for i in range(0, len(indices), self.batch_size):
                # Get the indices for this batch
                batch_indices = indices[i:i + self.batch_size]

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
                    if not self.validation_phase:
                        confmap, image = self.do_augmentations(confmap, image)

                    batch_images.append(image)
                    batch_confmaps.append(confmap)

                yield (np.array(batch_images), np.array(batch_confmaps))

    def generate_with_camera_matrices(self):
        while True:
            # Select indices for the batch
            if not self.validation_phase:
                batch_indices = np.random.choice(np.arange(len(self.box)), size=self.batch_size)
            else:
                batch_indices = np.arange(len(self.box))

            batch_images = []
            batch_cam_matrices = []
            batch_inv_camera_matrices = []
            batch_confmaps = []

            for idx in batch_indices:
                image = self.box[idx]
                points_3D = self.points_3D[idx]
                confmap = self.confmaps[idx]

                # Apply your augmentation function
                if not self.validation_phase:
                    confmap, image = self.do_augmentations(confmap, image)

                camera_matrices, inv_camera_matrices = self.get_camera_matrices(confmap, points_3D)

                batch_cam_matrices.append(camera_matrices)
                batch_inv_camera_matrices.append(inv_camera_matrices)
                batch_images.append(image)
                batch_confmaps.append(confmap)

            output = ((np.array(batch_images),
                       np.array(batch_cam_matrices),
                       np.array(batch_inv_camera_matrices)),

                      np.array(batch_confmaps))
            yield output

    def do_augmentations(self, confmap, image):
        do_horizontal_flip = bool(np.random.randint(2)) and self.do_horizontal_flip
        do_vertical_flip = bool(np.random.randint(2)) and self.do_vertical_flip
        rotation_angle = np.random.randint(-self.rotation_range, self.rotation_range)
        shift_y_x = np.random.randint(-self.xy_shifts, self.xy_shifts, 2)
        scaling = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        augmented_image = self.augment_image(image, do_horizontal_flip, do_vertical_flip, rotation_angle, shift_y_x,
                                             scaling)
        augmented_confmap = self.augment_image(confmap, do_horizontal_flip, do_vertical_flip, rotation_angle, shift_y_x,
                                               scaling)
        return augmented_confmap, augmented_image

    def augment_image(self, img, do_horizontal_flip, do_vertical_flip, rotation_angle, shift_y_x, scaling):
        num_channels = img.shape[-1]
        for channel in range(num_channels):
            img[:, :, channel] = Augmentor.augment(img[:, :, channel], do_horizontal_flip,
                                                   do_vertical_flip, rotation_angle, shift_y_x, scaling)
        return img

    def get_camera_matrices(self, confmap, points_3D):
        num_cameras = 4
        num_channels = confmap.shape[-1] // num_cameras
        confs_4 = []
        camera_matrices = []
        inv_camera_matrices = []

        points_2d = []
        for cam in range(num_cameras):
            confs_i = confmap[np.newaxis, ..., cam * num_channels + np.arange(num_channels)]
            points_2d_cam = self.tf_find_peaks(confs_i)
            points_2d.append(points_2d_cam)
            P = self.estimate_projection_matrix_dlt(points_3D, points_2d_cam)
            P_inv = np.linalg.pinv(P)
            camera_matrices.append(P)
            inv_camera_matrices.append(P_inv)

        cam1 = 0
        cam2 = 2
        P1 = camera_matrices[cam1]
        P2 = camera_matrices[cam2]
        points_2d_cam1 = points_2d[cam1]
        points_2d_cam2 = points_2d[cam2]

        self.check_error(P1, P2, points_2d_cam1, points_2d_cam2, points_3D)

        camera_matrices = np.array(camera_matrices)
        inv_camera_matrices = np.array(inv_camera_matrices)
        return camera_matrices, inv_camera_matrices

    @staticmethod
    def check_error(P1, P2, points_2d_cam1, points_2d_cam2, pts_3d):
        tr_points_3d = cv2.triangulatePoints(P1, P2, points_2d_cam1.T, points_2d_cam2.T).T
        tr_points_3d = tr_points_3d[:, :-1] / tr_points_3d[:, -1:]
        error = np.mean(np.abs(pts_3d - tr_points_3d))
        print('error: ', error)

    @staticmethod
    def estimate_projection_matrix_dlt(points_3d, points_2d):
        assert len(points_2d) == len(points_3d)
        assert len(points_2d) >= 6

        A = []

        for i in range(len(points_2d)):
            X, Y, Z = points_3d[i]
            x, y = points_2d[i]
            A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x])
            A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y])

        _, _, V = np.linalg.svd(A)
        P = V[-1].reshape(3, 4)
        P /= P[-1, -1]

        num_points = len(points_3d)
        points_3d_hom = np.column_stack((points_3d, np.ones((num_points, 1))))
        points_2d_reprojected_hom = np.dot(P, points_3d_hom.T).T
        points_2d_reprojected = points_2d_reprojected_hom[:, :-1] / points_2d_reprojected_hom[:, -1:]
        reprojection_error = np.mean(np.linalg.norm(points_2d_reprojected - points_2d, axis=-1))
        print('reprojection_error: ', reprojection_error)


        return P

    @staticmethod
    def tf_find_peaks(x):
        """ Finds the maximum value in each channel and returns the location and value.
        Args:
            x: rank-4 tensor (samples, height, width, channels)

        Returns:
            peaks: rank-3 tensor (samples, [x, y, val], channels)
        """

        # Store input shape
        in_shape = tf.shape(x)

        # Flatten height/width dims
        flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])

        # Find peaks in linear indices
        idx = tf.argmax(flattened, axis=1)

        # Convert linear indices to subscripts
        rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
        cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])

        # Dumb way to get actual values without indexing
        vals = tf.math.reduce_max(flattened, axis=1)

        # Return N x 3 x C tensor
        pred = tf.stack([
            tf.cast(cols, tf.float32),
            tf.cast(rows, tf.float32),
        ], axis=1)
        pred = np.squeeze(pred.numpy()).T
        return pred
