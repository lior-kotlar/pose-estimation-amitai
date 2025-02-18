
import numpy as np
import torchvision.transforms.functional as F
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Augmentor import Augmentor
import h5py
import cv2
from matplotlib import pyplot as plt
from constants import *
from scipy.linalg import null_space
# import wandb
np.random.seed(0)


class DataGenerator:
    def __init__(self, config, preprocessor):
        self.val_inds = None
        self.train_inds = None
        self.confmaps = None
        self.box = None
        self.vis_sample = None
        self.config = config
        self.model_type = self.config["model type"]
        self.val_fraction = config["val_fraction"]
        self.do_augmentations = self.config["do augmentations"]
        self.batch_size = config["batch_size"]
        self.preprocessor = preprocessor
        self.num_frames = self.preprocessor.get_num_frames()
        self.train_dataset, self.val_dataset = self.config_data_generator()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        # print(f"vals inds: {self.val_inds}")
        self.vis_sample = (self.box[self.val_inds[0]], self.confmaps[self.val_inds[0]])
        self.train_indices = np.arange(len(self.train_dataset))
        self.current_train_index = 0

    def shuffle_train_indices(self):
        np.random.shuffle(self.train_indices)
        self.current_train_index = 0

    def get_next_train_batch(self):
        batch_indices = []
        while len(batch_indices) < self.batch_size:
            remaining = self.batch_size - len(batch_indices)
            start_index = self.current_train_index
            end_index = start_index + remaining

            if end_index > len(self.train_indices):
                batch_indices.extend(self.train_indices[start_index:].tolist())
                self.current_train_index = 0
            else:
                batch_indices.extend(self.train_indices[start_index:end_index].tolist())
                self.current_train_index = end_index

        batch_indices = batch_indices[:self.batch_size]  # Ensure exact batch size

        # print(f"Current Train Index: {self.current_train_index}, Batch Indices: {batch_indices}")

        batch = [self.train_dataset[i] for i in batch_indices]
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])

        return inputs, targets

    def config_data_generator(self):
        if self.model_type == "ALL_CAMS_DISENTANGLED_PER_WING_CNN":
            self.box = self.preprocessor.get_box_orig()
            self.confmaps = self.preprocessor.get_confmaps_orig()
            self.num_samples = len(self.confmaps)
            self.train_inds, self.val_inds = self.get_train_val_split(self.num_samples)
            self.cropzone = self.preprocessor.get_cropzone()
            train_data_generator = CameraMatrixGenerator(self.config,
                                                         box=self.box[self.train_inds],
                                                         confmaps=self.confmaps[self.train_inds],
                                                         cropzone=self.cropzone[self.train_inds],
                                                         do_augmentations=self.do_augmentations)
            val_data_generator = CameraMatrixGenerator(self.config,
                                                       box=self.box[self.val_inds],
                                                       confmaps=self.confmaps[self.val_inds],
                                                       cropzone=self.cropzone[self.val_inds],
                                                       do_augmentations=False)
            return train_data_generator, val_data_generator

        else:
            self.box = self.preprocessor.get_box()
            self.confmaps = self.preprocessor.get_confmaps()
            self.num_samples = len(self.confmaps)
            self.train_inds, self.val_inds = self.get_train_val_split(self.num_samples)
            train_data_generator = DefaultDataset(self.config,
                                                  box=self.box[self.train_inds],
                                                  confmaps=self.confmaps[self.train_inds],
                                                  do_augmentations=self.do_augmentations)
            val_data_generator = DefaultDataset(self.config, box=self.box[self.val_inds],
                                                confmaps=self.confmaps[self.val_inds],
                                                do_augmentations=False)
            return train_data_generator, val_data_generator

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_val_dataloader(self):
        return self.val_dataloader

    def get_vis_sample(self):
        return self.vis_sample

    def get_train_val_split(self, num_samples):
        all_inds = np.arange(num_samples)
        np.random.shuffle(all_inds)
        val_size = round(num_samples * self.val_fraction)
        val_inds = all_inds[:val_size]
        train_inds = all_inds[val_size:]
        return train_inds, val_inds


class DefaultDataset(Dataset):
    def __init__(self, config, box, confmaps, do_augmentations=False):
        self.box = box
        self.confmaps = confmaps
        self.image_size = box.shape[-2]
        self.model_type = config["model type"]
        self.xy_shifts = config["augmentation shift x y"]
        self.rotation_range = config["rotation range"]
        self.do_horizontal_flip = bool(config["horizontal flip"])
        self.do_vertical_flip = bool(config["vertical flip"])
        self.scale_range = config["zoom range"]
        self.do_augmentations = do_augmentations
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.box)

    def __getitem__(self, idx):
        sample_box = self.box[idx]
        sample_confmaps = self.confmaps[idx]
        sample_box = self.transform(sample_box)
        sample_confmaps  = self.transform(sample_confmaps)
        if self.do_augmentations:
            if self.model_type == ALL_CAMS_18_POINTS:
                box1, box2, box3, box4 = np.array_split(sample_box, 4, axis=-1)
                confs1, confs2, confs3, confs4 = np.array_split(sample_confmaps, 4, axis=-1)

                box1, confs1 = self.augment_view(box1, confs1)
                box2, confs2 = self.augment_view(box2, confs2)
                box3, confs3 = self.augment_view(box3, confs3)
                box4, confs4 = self.augment_view(box4, confs4)

                sample_box = torch.cat((box1, box2, box3, box4), dim=-1)
                sample_confmaps = torch.cat((confs1, confs2, confs3, confs4), dim=-1)
            else:
                sample_box, sample_confmaps = self.cast_as_float(sample_box, sample_confmaps)

        # plt.imshow(sample_box[..., 1] + np.sum(sample_confmaps, axis=-1))
        # plt.show()
        sample_box, sample_confmaps = self.cast_as_float(sample_box, sample_confmaps)
        return sample_box, sample_confmaps

    def cast_as_float(self, sample_box, sample_confmaps):
        sample_box, sample_confmaps = self.augment_view(sample_box, sample_confmaps)
        try:
            sample_box, sample_confmaps = sample_box.float(), sample_confmaps.float()
        except TypeError as e:
            print(
                f"Error: {e} in line sample_box, sample_confmaps = sample_box.float(), sample_confmaps.float()")
        return sample_box, sample_confmaps

    def augment_view(self, sample_box, sample_confmaps):
        if self.rotation_range != 0:
            rotation_angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            rotation_angle = 0

        if self.xy_shifts != 0:
            shift_y = np.random.uniform(-self.xy_shifts, self.xy_shifts)
            shift_x = np.random.uniform(-self.xy_shifts, self.xy_shifts)
        else:
            shift_y = 0
            shift_x = 0

        do_horizontal_flip = np.random.rand() < 0.5 and self.do_horizontal_flip
        do_vertical_flip = np.random.rand() < 0.5 and self.do_vertical_flip
        scaling = np.random.uniform(self.scale_range[0], self.scale_range[1])
        # Apply augmentations to both sample_box and sample_confmaps
        sample_box = F.affine(sample_box, angle=rotation_angle, translate=(shift_x, shift_y),
                              scale=scaling, shear=0)
        sample_confmaps = F.affine(sample_confmaps, angle=rotation_angle, translate=(shift_x, shift_y),
                                   scale=scaling, shear=0)

        if do_horizontal_flip:
            sample_box = F.hflip(sample_box)
            sample_confmaps = F.hflip(sample_confmaps)

        if do_vertical_flip:
            sample_box = F.vflip(sample_box)
            sample_confmaps = F.vflip(sample_confmaps)

        return sample_box, sample_confmaps

    def augment_view_custom(self, sample_box, sample_confmaps):
        if self.rotation_range != 0:
            rotation_angle = np.random.randint(-self.rotation_range, self.rotation_range)
        else:
            rotation_angle = 0
        if self.xy_shifts != 0:
            shift_y_x = np.random.randint(-self.xy_shifts, self.xy_shifts, 2)
        else:
            shift_y_x = 0
        do_horizontal_flip = bool(np.random.randint(2)) and self.do_horizontal_flip
        do_vertical_flip = bool(np.random.randint(2)) and self.do_vertical_flip
        scaling = np.random.uniform(self.scale_range[0], self.scale_range[1])
        sample_box = Augmentor.custom_augmentations(np.copy(sample_box),
                                                    rotation_angle,
                                                    shift_y_x,
                                                    do_horizontal_flip,
                                                    do_vertical_flip,
                                                    scaling)
        sample_confmaps = Augmentor.custom_augmentations(np.copy(sample_confmaps),
                                                         rotation_angle,
                                                         shift_y_x,
                                                         do_horizontal_flip,
                                                         do_vertical_flip,
                                                         scaling)
        return sample_box, sample_confmaps


class CameraMatrixGenerator(Dataset):
    def __init__(self, config, box, confmaps, do_augmentations=False, cropzone=None):
        self.box = box
        self.confmaps = confmaps
        self.image_size = box.shape[-2]
        self.data_path = config["data_path"]
        self.cropzone = cropzone
        self.camera_matrices = h5py.File(self.data_path, "r")["cameras_dlt_array"][:].T
        self.get_camera_matrix_decomposition()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.box)

    def __getitem__(self, idx):
        sample_box = self.box[idx]
        sample_confmaps = self.confmaps[idx]
        cropzones = self.cropzone[idx]
        camera_matrices, inv_camera_matrices = self.get_cropped_camera_matrices(cropzones)

        # check calibration
        # self.check_cropped_camera_calibration(camera_matrices, sample_confmaps)

        sample_confmaps = self.prepare_confmaps(sample_confmaps)
        sample_box = [sample_box[..., [0, 1, 2, 3]], sample_box[..., [0, 1, 2, 4]]]

        # check calibration

        # choose wing at random
        wing = np.random.randint(2)

        sample_box = sample_box[wing]
        sample_confmaps = sample_confmaps[wing]

        sample_confmaps = np.squeeze(np.concatenate(np.array_split(sample_confmaps, 4, axis=0), axis=-1))
        sample_box = np.squeeze(np.concatenate(np.array_split(sample_box, 4, axis=0), axis=-1))

        sample_box, sample_confmaps = self.transform(sample_box).float(), self.transform(sample_confmaps).float()
        inputs = [sample_box,
                  torch.tensor(camera_matrices).float(),
                  torch.tensor(inv_camera_matrices).float()]
        targets = sample_confmaps
        return inputs, targets

    @staticmethod
    def prepare_confmaps(sample_confmaps):
        head_tail = sample_confmaps[..., [-2, -1]]
        wings = sample_confmaps[..., :-2]
        left_confs, right_confs = np.array_split(wings, 2, axis=-1)
        left_confs = np.concatenate((left_confs, head_tail), axis=-1)
        right_confs = np.concatenate((right_confs, head_tail), axis=-1)
        sample_confmaps = [left_confs, right_confs]
        return sample_confmaps

    def check_cropped_camera_calibration(self, cropped_camera_matrices, sample_confmaps):
        all = []
        for couple in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
            # couple = [2, 3]
            Ps = cropped_camera_matrices
            points_2D = Augmentor.tf_find_peaks(sample_confmaps)
            # move to camera coordinates
            points_2D = points_2D.copy()
            y = 0
            points_2D[..., 1] = 192 - points_2D[..., 1]

            a, b = couple
            Pa, Pb = Ps[couple]
            points_a = points_2D[a]
            points_b = points_2D[b]

            points_3D_exp = CameraMatrixGenerator.custom_triangulation(Pa, Pb, points_a, points_b)

            points_3d = cv2.triangulatePoints(Pa, Pb, points_a.T, points_b.T).T
            points_3d = points_3d[:, :-1] / points_3d[:, -1:]

            # print(np.abs(points_3d - points_3D_exp))
            # pass


            points_3d_h = np.column_stack((points_3d, np.ones((points_3d.shape[0], 1))))
            points_a_rep = (Pa @ points_3d_h.T).T
            points_a_rep = points_a_rep[:, :-1] / points_a_rep[:, -1:]

            points_b_rep = (Pb @ points_3d_h.T).T
            points_b_rep = points_b_rep[:, :-1] / points_b_rep[:, -1:]

            reprojection_errors_a = np.mean(np.linalg.norm(points_a - points_a_rep, axis=-1))
            reprojection_errors_b = np.mean(np.linalg.norm(points_b - points_b_rep, axis=-1))
            mean = np.mean([reprojection_errors_a, reprojection_errors_b])
            all.append(mean)
        print(np.mean(all))
        plt.hist(all, bins=30)
        plt.show()

    @staticmethod
    def custom_triangulation(Pa, Pb, points_a, points_b):
        N = points_a.shape[0]
        # Extract rows of Pa and Pb
        p1a, p2a, p3a = Pa[0, :], Pa[1, :], Pa[2, :]
        p1b, p2b, p3b = Pb[0, :], Pb[1, :], Pb[2, :]

        # Initialize the A matrix for all points
        A = np.zeros((N, 4, 4))

        # Fill the A matrix
        A[:, 0, :] = points_a[:, 0:1] * p3a - p1a
        A[:, 1, :] = points_a[:, 1:2] * p3a - p2a
        A[:, 2, :] = points_b[:, 0:1] * p3b - p1b
        A[:, 3, :] = points_b[:, 1:2] * p3b - p2b

        # Solve AX = 0 using SVD for each A
        X = np.zeros((N, 4))
        for i in range(N):
            _, _, Vt = np.linalg.svd(A[i])
            X[i] = Vt[-1]  # The last row of Vt corresponds to the solution
        # Normalize X to convert from homogeneous coordinates and exclude the last component
        X = X[:, :-1] / X[:, -1:]
        return X


    def check_calibration(self, cropzones, sample_confmaps):
        # sample_confmaps are of shape (4, 192, 192, num_points)
        all_points = Augmentor.tf_find_peaks(sample_confmaps)
        cam1, cam2, cam3, cam4 = all_points
        couple = [2, 3]
        points_2D = np.array([cam1, cam2, cam3, cam4])
        a, b = couple
        Pa, Pb = self.camera_matrices[couple]
        points_a = points_2D[a]
        crop_a = cropzones[a]
        xa = points_a[:, 0] + crop_a[1]
        ya = points_a[:, 1] + crop_a[0]
        ya = 801 - ya
        points_a = np.column_stack((xa, ya))
        points_b = points_2D[b]
        crop_b = cropzones[b]
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
        print(mean)

    def get_cropped_camera_matrices(self, cropzones):
        cropped_Ps = []
        cropped_invPs = []
        for cam in range(cropzones.shape[0]):
            y_crop, x_crop = cropzones[cam, :]
            K = self.Ks[cam]
            K /= K[-1, -1]
            dx = x_crop
            dy = 800 + 1 - y_crop - 192
            K_prime = K.copy()
            K_prime[0, 2] -= dx  # adjust x-coordinate of the principal point
            K_prime[1, 2] -= dy  # adjust y-coordinate of the principal point
            R = self.Rs[cam]
            t = self.ts[cam]
            P_prime = K_prime @ np.column_stack((R, t))
            P_prime /= np.linalg.norm(P_prime)
            cropped_Ps.append(P_prime)
            inv_P_prime = np.linalg.pinv(P_prime)
            inv_P_prime /= np.linalg.norm(inv_P_prime)
            cropped_invPs.append(inv_P_prime)
        return np.array(cropped_Ps), np.array(cropped_invPs)

    def get_camera_matrix_decomposition(self):
        self.Ks = []
        self.Rs = []
        self.ts = []
        for P in self.camera_matrices:
            K, Rc_w, Pc, pp, pv = DecomposeCamera(P)
            t = (-Rc_w @ Pc)[:, np.newaxis]
            self.Ks.append(K)
            self.Rs.append(Rc_w)
            self.ts.append(t)


def uncrop(cam_points, cropzones, cam_num):
    uncroped = []
    for p in range(len(cam_points)):
        x = cropzones[cam_num, 1] + cam_points[p, 0]
        y = cropzones[cam_num, 0] + cam_points[p, 1]
        y = 800 + 1 - y
        point = [x, y, 1]
        uncroped.append(point)
    return np.array(uncroped)


def RQ3(A):
    """
    RQ decomposition of 3x3 matrix

    :param A: 3x3 matrix
    :returns: R - Upper triangular 3x3 matrix, Q - 3x3 orthonormal rotation matrix
    """
    if A.shape != (3, 3):
        raise ValueError('A must be a 3x3 matrix')

    eps = 1e-10

    # Find rotation Qx to set A[2,1] to 0
    A[2, 2] = A[2, 2] + eps
    c = -A[2, 2] / np.sqrt(A[2, 2] ** 2 + A[2, 1] ** 2)
    s = A[2, 1] / np.sqrt(A[2, 2] ** 2 + A[2, 1] ** 2)
    Qx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    R = A @ Qx

    # Find rotation Qy to set A[2,0] to 0
    R[2, 2] = R[2, 2] + eps
    c = R[2, 2] / np.sqrt(R[2, 2] ** 2 + R[2, 0] ** 2)
    s = R[2, 0] / np.sqrt(R[2, 2] ** 2 + R[2, 0] ** 2)
    Qy = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    R = R @ Qy

    # Find rotation Qz to set A[1,0] to 0
    R[1, 1] = R[1, 1] + eps
    c = -R[1, 1] / np.sqrt(R[1, 1] ** 2 + R[1, 0] ** 2)
    s = R[1, 0] / np.sqrt(R[1, 1] ** 2 + R[1, 0] ** 2)
    Qz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = R @ Qz

    Q = Qz.T @ Qy.T @ Qx.T

    # Adjust R and Q so that the diagonal elements of R are +ve
    for n in range(3):
        if R[n, n] < 0:
            R[:, n] = -R[:, n]
            Q[n, :] = -Q[n, :]

    return R, Q


def DecomposeCamera(P):
    """
    Decomposes a camera projection matrix

    :param P: 3x4 camera projection matrix
    :returns: K, Rc_w, Pc, pp, pv
    """
    p1 = P[:, 0]
    p2 = P[:, 1]
    p3 = P[:, 2]
    p4 = P[:, 3]

    M = P[:, :3]
    m3 = M[2, :].T

    # Camera centre, analytic solution
    X = np.linalg.det(np.column_stack((p2, p3, p4)))
    Y = -np.linalg.det(np.column_stack((p1, p3, p4)))
    Z = np.linalg.det(np.column_stack((p1, p2, p4)))
    T = -np.linalg.det(M)

    Pc = np.array([X, Y, Z, T])
    Pc = Pc / Pc[3]
    Pc = Pc[:3]  # Make inhomogeneous

    # Principal point
    pp = M @ m3
    pp = pp / pp[2]
    pp = pp[:2]  # Make inhomogeneous

    # Principal vector pointing out of camera
    pv = np.linalg.det(M) * m3
    pv = pv / np.linalg.norm(pv)

    # Perform RQ decomposition of M matrix
    K, Rc_w = RQ3(M)

    # Check if Rc_w is right-handed
    if np.dot(np.cross(Rc_w[:, 0], Rc_w[:, 1]), Rc_w[:, 2]) < 0:
        print('Note that rotation matrix is left handed')

    return K, Rc_w, Pc, pp, pv
