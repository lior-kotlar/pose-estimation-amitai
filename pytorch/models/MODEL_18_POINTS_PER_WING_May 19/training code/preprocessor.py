import matplotlib.pyplot as plt
import numpy as np
import cv2
from constants import *
import h5py
# import tensorflow as tf
import torch
from scipy.ndimage import binary_dilation, binary_closing, distance_transform_edt
from Augmentor import Augmentor
WHICH_TO_FLIP = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).astype(bool)

class Preprocessor:
    def __init__(self, config):
        self.confmaps_orig = None
        self.box_orig = None
        self.data_path = config['data_path']
        self.test_path = config['test_path']
        self.mix_with_test = bool(config['mix_with_test'])
        self.model_type = config['model type']
        self.mask_dilation = config['mask dilation']
        self.debug_mode = bool(config['debug mode'])
        self.wing_size_rank = config["rank wing size"]
        self.do_curriculum_learning = config["do curriculum learning"]
        self.single_time_channel = bool(config["single time channel"])
        self.box, self.confmaps = self.load_dataset(self.data_path)
        self.num_frames = self.box.shape[0]
        self.num_channels = self.box.shape[-1]
        self.preprocess_function = self.get_preprocces_function()

        # get useful indexes
        self.num_dims = len(self.box.shape)
        self.num_channels = self.box.shape[-1]
        self.num_time_channels = self.num_channels - 2
        self.left_mask_ind = self.num_time_channels
        self.first_mask_ind = self.num_time_channels
        self.right_mask_ind = self.left_mask_ind + 1
        self.time_channels = np.arange(self.num_time_channels)
        self.fly_with_left_mask = np.append(self.time_channels, self.left_mask_ind)
        self.fly_with_right_mask = np.append(self.time_channels, self.right_mask_ind)

        self.num_samples = None
        if self.debug_mode:
            num_debug_inds = 10
            if self.num_dims == 5:
                self.box = self.box[:num_debug_inds, :, :, :, :]
                self.confmaps = self.confmaps[:num_debug_inds, :, :, :, :]
            else:
                self.box = self.box[:, :num_debug_inds, :, :, :, :]
                self.confmaps = self.confmaps[:, :num_debug_inds, :, :, :, :]
            self.num_frames = self.box.shape[0]
            self.mix_with_test = False
        self.retrieve_points_3D()
        self.retrive_cropzone()
        self.camera_matrices = h5py.File(self.data_path, "r")["cameras_dlt_array"][:].T

    def retrive_cropzone(self):
        self.cropzone = h5py.File(self.data_path, "r")["/cropZone"][:]
        self.cropzone_per_wing = np.repeat(self.cropzone, 2, axis=0)

    def retrieve_points_3D(self):
        self.points_3D = h5py.File(self.data_path, "r")["/points_3D"][:]
        self.points_3D = np.transpose(self.points_3D, [1, 2, 0])[:self.box.shape[0]]
        self.num_points = self.points_3D.shape[1]
        self.num_wing_points = self.num_points - 2
        self.left_inds = np.arange(0, self.num_wing_points // 2)
        self.right_inds = np.arange(self.num_wing_points // 2, self.num_wing_points)
        self.head_tail_inds = np.array([-2, -1])
        left_wing = self.points_3D[:, np.append(self.left_inds, self.head_tail_inds), :]
        right_wing = self.points_3D[:, np.append(self.right_inds, self.head_tail_inds), :]
        self.points_3D_per_wing = np.concatenate((left_wing, right_wing), axis=0)
        self.points_3D_per_camera = np.repeat(np.expand_dims(self.points_3D, axis=1), self.box.shape[1], axis=1)

    def get_box(self):
        return self.box

    def get_num_frames(self):
        return self.num_frames

    def get_confmaps(self):
        return self.confmaps

    def get_box_orig(self):
        return self.box_orig

    def get_confmaps_orig(self):
        return self.confmaps_orig

    def get_cropzone_per_wing(self):
        return self.cropzone_per_wing

    def get_cropzone(self):
        return self.cropzone

    def get_points_3D_per_wing(self):
        return self.points_3D_per_wing

    def do_preprocess(self):
        if self.mix_with_test:
            self.do_mix_with_test()
        self.preprocess_function()

    def load_dataset(self, data_path):
        """ Loads and normalizes datasets. """
        # Load
        X_dset = "box"
        Y_dset = "confmaps"
        with h5py.File(data_path, "r") as f:
            X = f[X_dset][:]
            Y = f[Y_dset][:]

        # Adjust dimensions
        X = self.preprocess(X, permute=None)
        Y = self.preprocess(Y, permute=None)
        if X.shape[0] != 2 and X.shape[1] != 4:
            X = X.T
        if Y.shape[0] != 2 or Y.shape[1] == 192:
            Y = Y.T
        return X, Y

    def get_preprocces_function(self):
        if self.model_type == ALL_POINTS_MODEL or self.model_type == ALL_POINTS_MODEL_VIT:
            return self.reshape_to_cnn_input
        elif self.model_type == PER_WING_MODEL:
            return self.do_reshape_per_wing
        elif self.model_type == TRAIN_ON_3_GOOD_CAMERAS_MODEL:
            return self.do_reshape_per_wing
        elif self.model_type == ALL_CAMS:
            return self.do_reshape_per_wing
        elif (self.model_type == MODEL_18_POINTS_PER_WING or self.model_type == MODEL_18_POINTS_3_GOOD_CAMERAS or
              self.model_type == MODEL_18_POINTS_PER_WING_VIT or self.model_type == GPTNET):
            return self.do_preprocess_18_pnts
        elif (self.model_type == ALL_CAMS_18_POINTS or self.model_type == ALL_CAMS_DISENTANGLED_PER_WING_VIT
              or self.model_type == ALL_CAMS_DISENTANGLED_PER_WING_CNN or self.model_type == ALL_CAMS_18_POINTS_VIT):
            return self.reshape_for_ALL_CAMS_18_POINTS

    def do_mix_with_test(self):
        test_box, test_confmaps = self.load_dataset(self.test_path)
        if test_box.shape[0] != 2:
            test_box = np.transpose(test_box, (5, 4, 3, 2, 1, 0))
            test_confmaps = np.transpose(test_confmaps, (5, 4, 3, 2, 1, 0))
        trainset_type = MOVIE_TRAIN_SET
        test_box[0], test_confmaps[0] = self.split_per_wing(test_box[0], test_confmaps[0], ALL_POINTS_MODEL, trainset_type)
        test_box[1], test_confmaps[1] = self.split_per_wing(test_box[1], test_confmaps[1], ALL_POINTS_MODEL, trainset_type)
        test_box[0], problematic_masks_inds = self.fix_movie_masks(test_box[0])
        test_box[1], problematic_masks_inds = self.fix_movie_masks(test_box[1])
        problematic_masks_inds = np.array(problematic_masks_inds)
        self.box = np.concatenate((self.box, test_box), axis=1)
        self.confmaps = np.concatenate((self.confmaps, test_confmaps), axis=1)
        self.num_frames = self.box.shape[1]

    def split_per_wing(self, box, confmaps, model_type, trainset_type):
        """ make sure the confmaps fits the wings1 """
        min_in_mask = 3
        num_joints = confmaps.shape[-1]
        num_joints_per_wing = int(num_joints / 2)
        LEFT_INDEXES = np.arange(0, num_joints_per_wing)
        RIGHT_INDEXES = np.arange(num_joints_per_wing, 2 * num_joints_per_wing)

        left_wing_box = box[:, :, :, :, self.fly_with_left_mask]
        right_wing_box = box[:, :, :, :, self.fly_with_right_mask]
        right_wing_confmaps = confmaps[:, :, :, :, LEFT_INDEXES]
        left_wing_confmaps = confmaps[:, :, :, :, RIGHT_INDEXES]

        num_frames = box.shape[0]
        num_cams = box.shape[1]
        num_pts_per_wing = right_wing_confmaps.shape[-1]
        left_peaks = np.zeros((num_frames, num_cams, 2, num_pts_per_wing))
        right_peaks = np.zeros((num_frames, num_cams, 2, num_pts_per_wing))
        for cam in range(num_cams):
            l_p = Preprocessor.tf_find_peaks(left_wing_confmaps[:, cam, :, :, :])[:, :2, :]
            r_p = Preprocessor.tf_find_peaks(right_wing_confmaps[:, cam, :, :, :])[:, :2, :]
            left_peaks[:, cam, :, :] = l_p
            right_peaks[:, cam, :, :] = r_p

        left_peaks = left_peaks.astype(int)
        right_peaks = right_peaks.astype(int)

        new_left_wing_box = np.zeros(left_wing_box.shape)
        new_right_wing_box = np.zeros(right_wing_box.shape)
        new_right_wing_confmaps = np.zeros(right_wing_confmaps.shape)
        new_left_wing_confmaps = np.zeros(left_wing_confmaps.shape)

        num_of_bad_masks = 0
        # fit confmaps to wings1
        num_frames = box.shape[0]
        for frame in range(num_frames):
            for cam in range(num_cams):
                append = True
                fly_image = left_wing_box[frame, cam, :, :, self.time_channels]

                left_confmap = left_wing_confmaps[frame, cam, :, :, :]
                right_confmap = right_wing_confmaps[frame, cam, :, :, :]

                left_mask = left_wing_box[frame, cam, :, :, self.first_mask_ind]
                right_mask = right_wing_box[frame, cam, :, :, self.first_mask_ind]

                left_peaks_i = left_peaks[frame, cam, :, :]
                right_peaks_i = right_peaks[frame, cam, :, :]

                # check peaks
                left_values = 0
                right_values = 0
                for i in range(left_peaks_i.shape[-1]):
                    left_values += left_mask[left_peaks_i[1, i], left_peaks_i[0, i]]
                    right_values += right_mask[right_peaks_i[1, i], right_peaks_i[0, i]]

                # switch train_masks if peaks are completely missed
                if left_values < min_in_mask and right_values < min_in_mask:
                    temp = left_mask
                    left_mask = right_mask
                    right_mask = temp

                # check peaks again
                left_values = 0
                right_values = 0
                for i in range(left_peaks_i.shape[-1]):
                    left_values += left_mask[left_peaks_i[1, i], left_peaks_i[0, i]]
                    right_values += right_mask[right_peaks_i[1, i], right_peaks_i[0, i]]

                # don't append if one mask is missing
                mask_exist = True
                # if left_values < min_in_mask or right_values < min_in_mask:
                #     mask_exist = False
                #     num_of_bad_masks += 1

                if trainset_type == MOVIE_TRAIN_SET or (trainset_type == RANDOM_TRAIN_SET and mask_exist):
                    # copy fly image
                    new_left_wing_box[frame, cam, :, :, self.time_channels] = fly_image
                    new_left_wing_box[frame, cam, :, :, self.first_mask_ind] = left_mask
                    # copy mask
                    new_right_wing_box[frame, cam, :, :, self.time_channels] = fly_image
                    new_right_wing_box[frame, cam, :, :, self.first_mask_ind] = right_mask
                    # copy confmaps
                    new_right_wing_confmaps[frame, cam, :, :, :] = right_confmap
                    new_left_wing_confmaps[frame, cam, :, :, :] = left_confmap

        # make sure there is 3D right left consistency
        # self.ensure_right_left_consistency(new_left_wing_box,
        #                               new_right_wing_box,
        #                               new_left_wing_confmaps,
        #                               new_right_wing_confmaps)

        # save the original box and confidence maps
        self.box_orig = np.zeros(list(new_left_wing_box.shape[:-1]) + [5])
        self.box_orig[..., [0, 1, 2, 3]] = new_left_wing_box
        self.box_orig[..., -1] = new_right_wing_box[..., -1]
        self.confmaps_orig = np.concatenate((new_left_wing_confmaps, new_right_wing_confmaps), axis=-1)

        # b = np.transpose(self.box_orig[0, 0, :, :, [1, 3, 4]], [1, 2, 0])
        # c = np.sum(self.confmaps_orig[0, 0, ...], axis=-1)[..., np.newaxis]
        # im = b + c
        # plt.imshow(im)
        # plt.show()

        if model_type == PER_WING_MODEL:
            box = np.concatenate((new_left_wing_box, new_right_wing_box), axis=0)
            confmaps = np.concatenate((new_left_wing_confmaps, new_right_wing_confmaps), axis=0)

        elif model_type == ALL_POINTS_MODEL:
            # copy fly
            box[:, :, :, :, self.time_channels] = new_left_wing_box[:, :, :, :, self.time_channels]
            # copy left mask
            box[:, :, :, :, self.left_mask_ind] = new_left_wing_box[:, :, :, :, self.first_mask_ind]
            box[:, :, :, :, self.right_mask_ind] = new_right_wing_box[:, :, :, :, self.first_mask_ind]
            confmaps[:, :, :, :, LEFT_INDEXES] = new_left_wing_confmaps
            confmaps[:, :, :, :, RIGHT_INDEXES] = new_right_wing_confmaps

        print(f"finish preprocess. number of bad train_masks = {num_of_bad_masks}")
        return box, confmaps

    def ensure_right_left_consistency(self, new_left_wing_box,
                                      new_right_wing_box,
                                      new_left_wing_confmaps,
                                      new_right_wing_confmaps):

        box = np.zeros(list(new_left_wing_box.shape[:-1]) + [5])
        box[..., [0,1,2,3]] = new_right_wing_box
        box[..., -1] = new_left_wing_box[..., -1]
        confmaps = np.concatenate((new_left_wing_confmaps, new_right_wing_confmaps), axis=-1)
        confmaps = np.reshape(confmaps, [-1] + list(confmaps.shape[2:])).astype(np.float64)
        all_2D_points = Augmentor.tf_find_peaks(confmaps)
        all_2D_points = np.reshape(all_2D_points, (all_2D_points.shape[0] // 4, 4, -1, 2))
        cameras_to_check = np.array([1, 2, 3])
        all_min_scores = []
        for frame in range(self.num_frames):
            num_of_options = len(WHICH_TO_FLIP)
            switch_scores = np.zeros(num_of_options, )
            for i, option in enumerate(WHICH_TO_FLIP):
                points_2D = all_2D_points[frame].copy()
                cropzone = self.cropzone[frame]
                cameras_to_flip = cameras_to_check[option]
                for cam in cameras_to_flip:
                    left_points = points_2D[cam, self.left_inds, :]
                    right_points = points_2D[cam, self.right_inds, :]
                    points_2D[cam, self.left_inds, :] = right_points
                    points_2D[cam, self.right_inds, :] = left_points
                score = self.get_reprojection_error(points_2D, cropzone)
                switch_scores[i] = score
            cameras_to_flip = cameras_to_check[WHICH_TO_FLIP[np.argmin(switch_scores)]]
            all_min_scores.append(switch_scores[0])
            # print(switch_scores)
            if cameras_to_flip != []:
                print(frame)

    def get_reprojection_error(self, points_2D, cropzone):
        all_couples = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        all_reprojection_errors = []
        for couple in all_couples:
            a, b = couple
            Pa, Pb = self.camera_matrices[couple]

            points_a = points_2D[a]
            crop_a = cropzone[a]
            xa = points_a[:, 0] + crop_a[1]
            ya = points_a[:, 1] + crop_a[0]
            ya = 801 - ya
            points_a = np.column_stack((xa, ya))

            points_b = points_2D[b]
            crop_b = cropzone[b]
            xb = points_b[:, 0] + crop_b[1]
            yb = points_b[:, 1] + crop_b[0]
            yb = 801 - yb
            points_b = np.column_stack((xb, yb))

            # triangulate
            points_3d = cv2.triangulatePoints(Pa, Pb, points_a.T, points_b.T).T
            points_3d = points_3d[:, :-1] / points_3d[:, -1:]

            # get reprojection
            points_3d_h = np.column_stack((points_3d, np.ones((points_3d.shape[0], 1))))

            points_a_rep = (Pa @ points_3d_h.T).T
            points_a_rep = points_a_rep[:, :-1] / points_a_rep[:, -1:]

            points_b_rep = (Pb @ points_3d_h.T).T
            points_b_rep = points_b_rep[:, :-1] / points_b_rep[:, -1:]

            reprojection_errors_a = np.mean(np.linalg.norm(points_a - points_a_rep, axis=-1))
            reprojection_errors_b = np.mean(np.linalg.norm(points_b - points_b_rep, axis=-1))
            mean = np.mean([reprojection_errors_a, reprojection_errors_b])
            all_reprojection_errors.append(mean)

        mean_error = np.mean(all_reprojection_errors)
        return mean_error


    def fix_movie_masks(self, box):
        """
        goes through each frame, if there is no mask for a specific wing, unite train_masks of the closest times before and after
        this frame.
        :param box: a box of size (num_frames, 20, 192, 192)
        :return: same box
        """
        search_range = 5
        num_channels = 5
        num_frames = int(box.shape[0])
        problematic_masks = []
        for frame in range(num_frames):
            for cam in range(4):
                for mask_num in range(2):
                    mask = box[frame, cam, :, :, self.num_time_channels + mask_num]
                    if np.all(mask == 0):  # check if all 0:
                        problematic_masks.append((frame, cam, mask_num))
                        # find previous matching mask
                        prev_mask = np.zeros(mask.shape)
                        next_mask = np.zeros(mask.shape)
                        for prev_frame in range(frame - 1, max(0, frame - search_range - 1), -1):
                            prev_mask_i = box[prev_frame, cam, :, :, self.num_time_channels + mask_num]
                            if not np.all(prev_mask_i == 0):  # there is a good mask
                                prev_mask = prev_mask_i
                                break
                        # find next matching mask
                        for next_frame in range(frame + 1, min(num_frames, frame + search_range)):
                            next_mask_i = box[next_frame, cam, :, :, self.num_time_channels + mask_num]
                            if not np.all(next_mask_i == 0):  # there is a good mask
                                next_mask = next_mask_i
                                break
                        # combine the 2 train_masks
                        new_mask = prev_mask + next_mask
                        new_mask[new_mask >= 1] = 1
                        # replace empty mask with new mask
                        box[frame, cam, :, :, self.num_time_channels + mask_num] = new_mask
                        # matplotlib.use('TkAgg')
                        # plt.imshow(new_mask)
                        # plt.show()

        return box, problematic_masks

    def adjust_mask(self, mask):
        mask = binary_closing(mask).astype(int)
        mask = binary_dilation(mask, iterations=self.mask_dilation).astype(int)
        return mask

    def adjust_masks_size_ALL_POINTS(self):
        for frame in range(self.num_samples):
            mask_1 = self.box[frame, :, :, self.right_mask_ind]
            mask_2 = self.box[frame, :, :, self.left_mask_ind]
            adjusted_mask_1 = self.adjust_mask(mask_1)
            adjusted_mask_2 = self.adjust_mask(mask_2)
            self.box[frame, :, :, self.right_mask_ind] = adjusted_mask_1
            self.box[frame, :, :, self.left_mask_ind] = adjusted_mask_2

    def reshape_to_cnn_input(self):
        """ reshape the  input from """
        # confmaps = np.transpose(confmaps, (5,4,3,2,1,0))
        head_tail_confmaps = self.confmaps[..., -2:]
        wings_confmaps = self.confmaps[..., :-2]
        self.box, wings_confmaps = self.split_per_wing(self.box, wings_confmaps, ALL_POINTS_MODEL, RANDOM_TRAIN_SET)
        self.confmaps = np.concatenate((wings_confmaps, head_tail_confmaps), axis=-1)
        self.box = np.reshape(self.box, [-1, self.box.shape[2], self.box.shape[3], self.box.shape[4]])
        self.confmaps = np.reshape(self.confmaps,
                              [-1, self.confmaps.shape[2], self.confmaps.shape[3], self.confmaps.shape[4]])
        self.num_samples = self.box.shape[0]
        self.adjust_masks_size_ALL_POINTS()

    def adjust_masks_size_per_wing(self):
        num_frames = self.box.shape[0]
        num_cams = self.box.shape[1]
        for frame in range(num_frames):
            for cam in range(num_cams):
                mask = self.box[frame, cam, :, :, self.first_mask_ind]
                adjusted_mask = self.adjust_mask(mask)
                self.box[frame, cam, :, :, self.first_mask_ind] = adjusted_mask

    @staticmethod
    def take_n_good_cameras(box, confmaps, n, wing_size_rank=3):
        num_frames = box.shape[0]
        num_cams = box.shape[1]
        new_num_cams = n
        image_shape = box.shape[2]
        num_channels_box = box.shape[-1]
        num_channels_confmap = confmaps.shape[-1]
        new_box = np.zeros((num_frames, new_num_cams, image_shape, image_shape, num_channels_box))
        new_confmap = np.zeros((num_frames, new_num_cams, image_shape, image_shape, num_channels_confmap))
        small_wings_box = np.zeros((num_frames, image_shape, image_shape, num_channels_box))
        small_wings_confmaps = np.zeros((num_frames, image_shape, image_shape, num_channels_confmap))
        d_size_wings_inds = np.zeros((num_frames,))
        for frame in range(num_frames):
            wings_size = np.zeros(num_cams)
            for cam in range(num_cams):
                wing_mask = box[frame, cam, :, :, -1]
                wings_size[cam] = np.count_nonzero(wing_mask)
            wings_size_argsort = np.argsort(wings_size)[::-1]
            d_size_wing_ind = wings_size_argsort[wing_size_rank]
            d_size_wings_inds[frame] = d_size_wing_ind
            best_n_cameras = np.sort(wings_size_argsort[:new_num_cams])
            new_box[frame, ...] = box[frame, best_n_cameras, ...]
            new_confmap[frame, ...] = confmaps[frame, best_n_cameras, ...]
            small_wings_box[frame, ...] = box[frame, d_size_wing_ind, ...]
            small_wings_confmaps[frame, ...] = confmaps[frame, d_size_wing_ind, ...]
        return new_box, new_confmap, small_wings_box, small_wings_confmaps, d_size_wings_inds.astype(int)

    def reshape_for_ALL_CAMS_18_POINTS(self):
        num_cams = self.box.shape[1]
        head_tail_confmaps = self.confmaps[..., -2:]
        num_of_frames = head_tail_confmaps.shape[0]
        wings_confmaps = self.confmaps[..., :-2]
        self.box, wings_confmaps = self.split_per_wing(self.box, wings_confmaps, PER_WING_MODEL, RANDOM_TRAIN_SET)
        left_confmaps = wings_confmaps[:num_of_frames]
        right_confmaps = wings_confmaps[num_of_frames:]
        left_confmaps = np.concatenate((left_confmaps, head_tail_confmaps), axis=-1)
        right_confmaps = np.concatenate((right_confmaps, head_tail_confmaps), axis=-1)
        self.confmaps = np.concatenate((left_confmaps, right_confmaps), axis=0)
        self.confmaps_orig = np.concatenate((self.confmaps_orig, head_tail_confmaps), axis=-1)
        self.adjust_masks_size_per_wing()

        cam_boxes = []
        cam_confmaps = []
        for cam in range(num_cams):
            box_cam_i = self.box[:, cam, :, :, :]
            cam_confmaps_i = self.confmaps[:, cam, :, :, :]
            cam_boxes.append(box_cam_i)
            cam_confmaps.append(cam_confmaps_i)
        self.box = np.concatenate(cam_boxes, axis=-1)
        self.confmaps = np.concatenate(cam_confmaps, axis=-1)

    def reshape_for_ALL_CAMS(self):
        image_size = self.confmaps.shape[-2]
        num_channels_img = self.box.shape[-1]
        num_channels_confmap = self.confmaps.shape[-1]
        num_cams = self.box.shape[1]

        # box = box.reshape([-1, image_size, image_size, num_channels_img])
        # confmaps = confmaps.reshape([-1, image_size, image_size, num_channels_confmap])
        self.box = self.box.reshape([-1, num_cams, image_size, image_size, num_channels_img])
        self.confmaps = self.confmaps.reshape([-1, num_cams, image_size, image_size, num_channels_confmap])

        cam_boxes = []
        cam_confmaps = []
        for cam in range(num_cams):
            box_cam_i = self.box[:, cam, :, :, :]
            cam_confmaps_i = self.confmaps[:, cam, :, :, :]
            cam_boxes.append(box_cam_i)
            cam_confmaps.append(cam_confmaps_i)
        self.box = np.concatenate(cam_boxes, axis=-1)
        self.confmaps = np.concatenate(cam_confmaps, axis=-1)

    def do_reshape_per_wing(self):
        """ reshape input to a per wing model input """
        if self.num_dims == 6:
            box_0, confmaps_0 = self.split_per_wing(self.box[0], self.confmaps[0], PER_WING_MODEL, RANDOM_TRAIN_SET)
            box_1, confmaps_1 = self.split_per_wing(self.box[1], self.confmaps[1], PER_WING_MODEL, RANDOM_TRAIN_SET)
            self.box = np.concatenate((box_0, box_1), axis=0)
            self.adjust_masks_size_per_wing()
            self.confmaps = np.concatenate((confmaps_0, confmaps_1), axis=0)
        else:
            self.box, self.confmaps = self.split_per_wing(self.box, self.confmaps, PER_WING_MODEL, RANDOM_TRAIN_SET)
            self.adjust_masks_size_per_wing()
        if self.model_type == TRAIN_ON_3_GOOD_CAMERAS_MODEL:
            n = 3 if self.model_type == TRAIN_ON_3_GOOD_CAMERAS_MODEL else 2
            self.box, self.confmaps, _, _, _ = self.take_n_good_cameras(self.box, self.confmaps, n)
        if (self.model_type == ALL_CAMS or self.model_type  == ALL_CAMS_DISENTANGLED_PER_WING_VIT):
            self.box, self.confmaps,  _, _, _ = self.take_n_good_cameras(self.box, self.confmaps, 4)
            self.reshape_for_ALL_CAMS()
            self.num_samples = self.box.shape[0]
            return
        else:
            self.box = np.reshape(self.box, newshape=[self.box.shape[0] * self.box.shape[1],
                                                      self.box.shape[2], self.box.shape[3],
                                                      self.box.shape[4]])
            self.confmaps = np.reshape(self.confmaps,
                                  newshape=[self.confmaps.shape[0] * self.confmaps.shape[1],
                                            self.confmaps.shape[2], self.confmaps.shape[3],
                                            self.confmaps.shape[4]])
        self.num_samples = self.box.shape[0]
        if self.do_curriculum_learning:
            self.do_sort_by_wing_size()

    def do_sort_by_wing_size(self):
        indexes = sorted(range(len(self.box)),
                         key=lambda i: -np.sum(np.count_nonzero(np.logical_and(self.box[i, :, :, 3],
                                                                               self.box[i, :, :, 1]),
                                                                               axis=1)))
        self.box = self.box[indexes]
        self.confmaps = self.confmaps[indexes]
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('TkAgg')
        # frame=119
        # plt.imshow(self.box[frame,:,:,1] + self.box[frame,:,:,3] + np.sum(self.confmaps[frame, :, :, :], axis=-1))
        # plt.show()

    @staticmethod
    def get_distance_from_point_to_mask(point_2D, mask):
        dist_transform = distance_transform_edt(np.logical_not(mask).astype(int))
        # Find the distance from the point to the nearest point in the mask
        distance = dist_transform[point_2D[1]][point_2D[0]]
        return distance

    def reshape_to_body_parts(self):
        """
        make sure that right point corresponds to right mask and same with the left
        and also train with 5 channels model
        """
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('TkAgg')

        _box = np.reshape(self.box, [self.box.shape[0] * self.box.shape[1] * self.box.shape[2], self.box.shape[3], self.box.shape[4], self.box.shape[5]])
        _confmaps = np.reshape(self.confmaps, [self.confmaps.shape[0] * self.confmaps.shape[1] * self.confmaps.shape[2], self.confmaps.shape[3],
                                          self.confmaps.shape[4], self.confmaps.shape[5]])
        num_images = _box.shape[0]
        peaks = self.tf_find_peaks(_confmaps[:, :, :, :])[:, :2, :]
        left = 1
        right = 2
        for img in range(num_images):
            left_mask = _box[img, :, :, 2 + left]
            right_mask = _box[img, :, :, 2 + right]
            left_peak = peaks[img, :, 0].astype(int)
            right_peak = peaks[img, :, 1].astype(int)
            dist_r2r = self.get_distance_from_point_to_mask(right_peak, right_mask)
            dist_l2r = self.get_distance_from_point_to_mask(left_peak, right_mask)
            dist_l2l = self.get_distance_from_point_to_mask(left_peak, left_mask)
            dist_r2l = self.get_distance_from_point_to_mask(right_peak, left_mask)

            if dist_r2r > dist_l2r and dist_l2l > dist_r2l:
                # switch
                _box[img, :, :, 2 + left] = right_mask
                _box[img, :, :, 2 + right] = left_mask

            # fig, ax = plt.subplots()
            # ax.imshow(right_mask)
            # # ax.scatter(left_peak[0], left_peak[1], color='red', s=40)
            # ax.scatter(right_peak[0], right_peak[1], color='red', s=40)
            # plt.show()
        self.box, self.confmaps = _box, _confmaps
        self.num_samples = self.box.shape[0]

    def do_preprocess_18_pnts(self):
        head_tail_confmaps = self.confmaps[..., -2:]
        num_of_frames = head_tail_confmaps.shape[0]
        wings_confmaps = self.confmaps[..., :-2]
        self.box, wings_confmaps = self.split_per_wing(self.box, wings_confmaps, PER_WING_MODEL, RANDOM_TRAIN_SET)
        left_confmaps = wings_confmaps[:num_of_frames]
        right_confmaps = wings_confmaps[num_of_frames:]
        left_confmaps = np.concatenate((left_confmaps, head_tail_confmaps), axis=-1)
        right_confmaps = np.concatenate((right_confmaps, head_tail_confmaps), axis=-1)
        self.confmaps = np.concatenate((left_confmaps, right_confmaps), axis=0)
        self.adjust_masks_size_per_wing()
        if self.model_type == MODEL_18_POINTS_3_GOOD_CAMERAS:
            self.box, self.confmaps, _, _, _ = self.take_n_good_cameras(self.box, self.confmaps, 3)
        self.box = np.reshape(self.box, newshape=[self.box.shape[0] * self.box.shape[1],
                                                  self.box.shape[2], self.box.shape[3],
                                                  self.box.shape[4]])
        self.confmaps = np.reshape(self.confmaps,
                                   newshape=[self.confmaps.shape[0] * self.confmaps.shape[1],
                                             self.confmaps.shape[2], self.confmaps.shape[3],
                                             self.confmaps.shape[4]])
        self.num_samples = self.box.shape[0]

    @staticmethod
    def preprocess(X, permute=(0, 3, 2, 1)):
        """ Normalizes input data. """

        # Add singleton dim for single train_images
        if X.ndim == 3:
            X = X[None, ...]

        # Adjust dimensions
        if permute != None:
            X = np.transpose(X, permute)

        # Normalize
        if X.dtype == "uint8" or np.max(X) > 1:
            X = X.astype("float32") / 255

        return X

    @staticmethod
    def tf_find_peaks(x):
        """ Finds the maximum value in each channel and returns the location and value.
        Args:
            x: rank-4 tensor (samples, height, width, channels)

        Returns:
            peaks: rank-3 tensor (samples, [x, y, val], channels)
        """

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

        return pred.detach().cpu().numpy()
