# CallBacks.py
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback, Callback
from scipy.io import savemat
import tensorflow as tf
from viz import show_pred, show_confmap_grid, plot_history
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import csv

class LossHistory(Callback):
    def __init__(self, run_path):
        super().__init__()
        self.run_path = run_path

    def on_train_begin(self, logs=None):
        self.history = []
        self.csv_file_path = os.path.join(self.run_path, "history.csv")
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'loss', 'val_loss'])

    def on_epoch_end(self, epoch, logs=None):
        self.history.append(logs.copy())
        savemat(os.path.join(self.run_path, "history.mat"),
                {k: [x[k] for x in self.history] for k in self.history[0].keys()})

        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, logs['loss'], logs.get('val_loss')])

        plot_history(self.history, save_path=os.path.join(self.run_path, "history.png"))


def tf_find_peaks(x):
    in_shape = tf.shape(x)
    flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])
    idx = tf.argmax(flattened, axis=1)
    rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
    cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])
    vals = tf.math.reduce_max(flattened, axis=1)
    pred = tf.stack([
        tf.cast(cols, tf.float32),
        tf.cast(rows, tf.float32),
    ], axis=1)
    return pred

class L2LossCallback(Callback):
    def __init__(self, validation_data, run_path):
        super().__init__()
        self.validation_data = validation_data
        self.run_path = run_path

    def on_epoch_end(self, epoch, logs=None):
        pred_peaks = tf_find_peaks(self.model.predict(self.validation_data[0]))
        gt_peaks = tf_find_peaks(self.validation_data[1])
        l2_distances = tf.sqrt(tf.reduce_sum(tf.square(pred_peaks - gt_peaks), axis=1))
        l2_loss = tf.reduce_mean(l2_distances)
        std = np.std(l2_distances.numpy())
        logs['val_l2_loss'] = l2_loss.numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(l2_distances.numpy(), bins=30, alpha=0.75)
        plt.title(f'L2 Distance Histogram - Epoch {epoch + 1} \nValidation L2 loss: {l2_loss.numpy()} std: {std}')
        plt.xlabel('L2 Distance')
        plt.ylabel('Frequency')
        histogram_path = os.path.join(self.run_path, "histograms", f"l2_histogram_epoch_{epoch + 1}.png")
        plt.savefig(histogram_path)
        plt.close()

class L2PerPointLossCallback(Callback):
    def __init__(self, validation_data, run_path):
        super().__init__()
        self.box, self.confmaps = validation_data
        self.run_path = run_path

    def on_epoch_end(self, epoch, logs=None, n_bins=20):
        pred_peaks = tf_find_peaks(self.model.predict(self.box))
        gt_peaks = tf_find_peaks(self.confmaps)
        pred_peaks = pred_peaks.numpy()
        gt_peaks = gt_peaks.numpy()
        pred_peaks = np.transpose(pred_peaks, (0, 2, 1))
        gt_peaks = np.transpose(gt_peaks, (0, 2, 1))
        num_joints = gt_peaks.shape[1]
        l2_per_point_dists = np.linalg.norm(pred_peaks - gt_peaks, axis=-1).T
        if num_joints > 20:
            cam1, cam2, cam3, cam4 = np.array_split(l2_per_point_dists, 4)
            l2_per_point_dists = np.concatenate((cam1, cam2, cam3, cam4), axis=1)
        num_points = l2_per_point_dists.shape[0]
        histogram_path = os.path.join(self.run_path, 'l2_histograms_per_point', f'validation_epoch_{epoch + 1}.png')
        fig, axs = plt.subplots(num_points, 1, figsize=(12, 4 * num_points))
        for i in range(num_points):
            ax = axs[i]
            ax.hist(l2_per_point_dists[i], bins=n_bins, edgecolor='black')
            mean_val = np.mean(l2_per_point_dists[i])
            std_val = np.std(l2_per_point_dists[i])
            ax.set_title(f'Histogram for Point {i + 1} - Mean: {mean_val:.2f}, Std: {std_val:.2f}', fontsize=12)
            ax.set_xlabel('L2 distance in pixels', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
        plt.tight_layout(pad=3.0)
        plt.savefig(histogram_path)
        plt.close(fig)

class CallBacks:
    def __init__(self, config, run_path, model, viz_sample, validation):
        self.model = model
        self.history_callback = LossHistory(run_path=run_path)
        self.reduce_lr_factor = config["reduce_lr_factor"]
        self.reduce_lr_patience = config["reduce_lr_patience"]
        self.reduce_lr_min_delta = config["reduce_lr_min_delta"]
        self.reduce_lr_cooldown = config["reduce_lr_cooldown"]
        self.reduce_lr_min_lr = config["reduce_lr_min_lr"]
        self.save_every_epoch = bool(config["save_every_epoch"])
        self.data_path = config["data_path"]
        self.camera_matrices = h5py.File(self.data_path, "r")['/cameras_dlt_array'][:].T
        self.l2_loss_callback = L2LossCallback(validation, run_path)
        self.l2_per_point_callback = L2PerPointLossCallback(validation, run_path)
        self.reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=self.reduce_lr_factor,
                                                    patience=self.reduce_lr_patience, verbose=1, mode="auto",
                                                    min_delta=self.reduce_lr_min_delta, cooldown=self.reduce_lr_cooldown,
                                                    min_lr=self.reduce_lr_min_lr)
        if self.save_every_epoch:
            self.checkpointer = ModelCheckpoint(
                filepath=os.path.join(run_path, "weights/weights.{epoch:03d}-{val_loss:.9f}.h5"),
                verbose=1, save_best_only=False)
        else:
            self.checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "best_model.h5"), verbose=1,
                                                save_best_only=True)
        self.viz_grid_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: show_confmap_grid(self.model, *viz_sample, plot=True,
                                                               save_path=os.path.join(
                                                                   run_path,
                                                                   "viz_confmaps/confmaps_%03d.png" % epoch),
                                                               show_figure=False))
        self.viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(self.model, *viz_sample,
                                                                                           save_path=os.path.join(
                                                                                               run_path,
                                                                                               "viz_pred", "pred_%03d.png" % epoch),
                                                                                           show_figure=False))

    def get_history_callback(self):
        return self.history_callback

    def get_callbacks(self):
        return [self.reduce_lr_callback,
                self.checkpointer,
                self.history_callback,
                self.viz_pred_callback,
                self.l2_loss_callback,
                self.l2_per_point_callback]
