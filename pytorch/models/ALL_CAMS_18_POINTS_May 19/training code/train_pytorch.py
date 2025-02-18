from constants import *
import json
import preprocessor
import os
import shutil
from datetime import date
from time import time
import sys
import numpy as np
import Network
import torch
import copy
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler
from PIL import Image
from Augmentor import Augmentor
import matplotlib.pyplot as plt
import csv
from Datagenerators import DefaultDataset, CameraMatrixGenerator, DataGenerator
from tqdm import tqdm
import CNNs
from utils import tf_find_peaks_argmax, find_peaks_soft_argmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast


try:
    torch.zeros(4).cuda()
except:
    print("No GPU found, doesnt use cuda")


# import wandb
np.random.seed(0)


class Trainer:
    def __init__(self, configuration_path):
        with open(configuration_path) as C:
            config = json.load(C)
            self.config = config
            self.batch_size = config['batch_size']
            self.num_epochs = config['epochs']
            self.batches_per_epoch = config['batches per epoch']
            self.val_fraction = config['val_fraction']
            self.debug_mode = bool(config["debug mode"])
            self.accumulation_steps = config['accumulation_steps']
            if self.debug_mode:
                self.val_fraction = 0.5
            self.base_output_path = config["base output path"]
            self.do_augmentations = bool(config["do augmentations"])
            self.viz_idx = 1
            self.loss_function = config["loss_function"]
            self.clean = bool(config["clean"])
            self.model_type = config["model type"]


            # Check if GPU is available and set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                torch.zeros(4).cuda()
            except:
                print("No GPU found, doesnt use cuda")
            if torch.cuda.is_available():
                print("**************** CUDA is available. Using GPU. ****************", flush=True)
            else:
                print("**************** CUDA is not available. Using CPU. ****************", flush=True)

            self.preprocessor = preprocessor.Preprocessor(config)
            print("doing preprocess", flush=True)
            self.preprocessor.do_preprocess()
            print("finished preprocess", flush=True)
            self.data_generator = DataGenerator(config, self.preprocessor)

        # create the running folders
        self.run_name = f"{self.model_type}_{date.today().strftime('%b %d')}"
        print(self.run_name, flush=True)
        self.run_path = self.create_run_folders()
        self.save_configuration()

        # do preprocessing according to the model type
        self.box, self.confmaps = self.preprocessor.get_box(), self.preprocessor.get_confmaps()

        # get the right model architecture
        self.img_size = self.box.shape[1:]
        self.num_output_channels = self.confmaps.shape[-1]
        self.network = Network.Network(config,
                                       image_size=self.img_size,
                                       num_output_channels=self.num_output_channels)
        self.model = self.network.get_model()
        # split to train and validation
        print("img_size:", self.img_size, flush=True)
        print("num_output_channels:", self.num_output_channels, flush=True)
        self.train_dataloader = self.data_generator.get_train_dataloader()
        self.val_dataloader = self.data_generator.get_val_dataloader()
        self.viz_sample = self.data_generator.get_vis_sample()

    def train(self):
        t0_train = time()
        print("Using device", self.device, flush=True)
        self.model = self.model.to(self.device)
        num_epochs = self.num_epochs
        best_loss = float('inf')
        train_losses = []
        val_losses = []
        l2_losses = []
        l2_losses_per_point = []
        l2_stds = []
        l2_max_outlier = []
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                           verbose=True, threshold=1e-5, threshold_mode='rel',
                                           cooldown=0, min_lr=1e-10)
        scaler = GradScaler()
        torch.manual_seed(0)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}", flush=True)
            running_loss = 0.0
            self.model.train()
            self.data_generator.shuffle_train_indices()

            for batch_num in range(self.batches_per_epoch):
                inputs, targets = self.data_generator.get_next_train_batch()
                batch_size = targets.size(0)
                if batch_num % 10 == 0:
                    print(f"Batch number is {batch_num + 1}, batch size is {batch_size}", flush=True)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                with autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / self.accumulation_steps  # Scale the loss

                scaler.scale(loss).backward()  # Scale and backpropagate the loss

                if (batch_num + 1) % self.accumulation_steps == 0:
                    scaler.step(optimizer)  # Update parameters
                    scaler.update()  # Update the scale for next iteration
                    optimizer.zero_grad()  # Reset gradients

                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / (self.batches_per_epoch * self.batch_size)
            print(f'Train Loss: {epoch_loss:.7f}', flush=True)
            train_losses.append(epoch_loss)

            self.model.eval()
            val_running_loss = 0.0
            l2_all_distances = np.zeros(0)
            l2_per_point_dists = []

            for i, (inputs, confmaps) in enumerate(self.val_dataloader):
                with torch.no_grad():
                    if isinstance(inputs, list):
                        inputs = [input_i.to(self.device) for input_i in inputs]
                        outputs = self.model(*inputs)
                    else:
                        inputs = inputs.to(self.device)
                        outputs = self.model(inputs)
                    confmaps = confmaps.to(self.device)
                    loss = criterion(outputs, confmaps)

                val_running_loss += loss.item() * outputs.size(0)
                dists_flatten, dists_per_point = self.find_l2_val_loss(outputs, confmaps)
                l2_all_distances = np.concatenate((l2_all_distances, dists_flatten))
                if dists_per_point.ndim == 2:
                    l2_per_point_dists.append(dists_per_point)

            l2_per_point_dists = np.concatenate(l2_per_point_dists, axis=1)
            epoch_loss = val_running_loss / len(self.val_dataloader.dataset)
            print(f'Val Loss: {epoch_loss:.4f}', flush=True)
            self.scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_model_path = os.path.join(self.run_path, 'best_model.pth')
                model_scripted = torch.jit.script(self.model)
                model_scripted.save(save_model_path)

            l2_stds.append(np.std(l2_all_distances))
            l2_losses.append(np.mean(l2_all_distances))
            l2_losses_per_point.append(l2_per_point_dists)
            l2_max_outlier.append(np.max(l2_all_distances))
            val_losses.append(epoch_loss)

            self.save_validation_image(epoch, self.model)
            self.save_checkpoint(epoch, epoch_loss, self.model, optimizer)
            self.save_losses_to_csv(epoch, train_losses, val_losses, l2_losses, l2_stds, l2_max_outlier)
            self.save_training_fig(epoch, train_losses, val_losses)
            self.save_l2_histogram(epoch, l2_all_distances)
            self.save_l2_histogram_per_point(epoch, l2_per_point_dists)

        elapsed_train = time() - t0_train
        print("Total runtime first loss: %.1f mins" % (elapsed_train / 60), flush=True)

    @staticmethod
    def find_l2_val_loss(output_confmaps, input_confmaps):
        output_points = Trainer.get_points_from_confmaps(output_confmaps)
        input_points = Trainer.get_points_from_confmaps(input_confmaps)
        dists_per_point = np.linalg.norm(output_points - input_points, axis=-1).T
        dists_flatten = dists_per_point.flatten()
        return dists_flatten, dists_per_point

    @staticmethod
    def get_points_from_confmaps(confmaps):
        confmaps = confmaps.detach().cpu().numpy()
        confmaps = np.transpose(confmaps, [0, 2, 3, 1])
        output_points = np.squeeze(Trainer.find_points(confmaps))
        # output_points = np.reshape(output_points, [-1, 2])
        return output_points

    def display_batch(self, inputs):
        image = inputs[0, ...]
        image = image.detach().cpu().numpy()
        image = np.transpose(image, [1, 2, 0])
        plt.imshow(image[..., [0, 1, 2]])
        plt.show()

    def save_validation_image(self, epoch, model):
        data_iter = iter(self.val_dataloader)
        (inputs, confmaps) = next(data_iter)
        if isinstance(inputs, list):
            inputs = [input_i.to(self.device) for input_i in inputs]
            output = model(*inputs).detach().cpu().numpy()[:1]
            image = np.transpose(inputs[0].detach().cpu().numpy()[0], [1, 2, 0])
        else:
            inputs = inputs.to(self.device)
            output = model(inputs).detach().cpu().numpy()[:1]
            image = np.transpose(inputs.detach().cpu().numpy()[0], [1, 2, 0])
        output = np.transpose(output, [0, 2, 3, 1])
        points = np.squeeze(self.find_points(output))
        if output.shape[-1] > 20:
            points_each_cam = np.array_split(points, 4)
            images = np.array_split(image, 4, axis=-1)
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs = axs.ravel()
            for i, (image, points) in enumerate(zip(images, points_each_cam)):
                axs[i].imshow(image[..., 1] + 0.5 * image[..., -1])
                axs[i].scatter(points_each_cam[i][:, 0], points_each_cam[i][:, 1], color='red', s=10, marker='o')
                axs[i].axis('off')
        else:
            plt.figure()
            plt.imshow(image[..., 1] + 0.5 * image[..., -1])
            plt.scatter(points[:, 0], points[:, 1], color='red', s=10, marker='o')

        image_path = os.path.join(self.run_path, 'viz_pred', f'validation_epoch_{epoch + 1}.png')
        plt.savefig(image_path)
        plt.close()

    def save_checkpoint(self, epoch, epoch_loss, model, optimizer):
        save_path = os.path.join(self.run_path, 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, save_path)

    def save_losses_to_csv(self, epoch, train_losses, val_losses, l2_losses, l2_stds, l2_max_outlier):
        csv_save_path = os.path.join(self.run_path, 'losses.csv')  # Path for the CSV file

        # Helper function to format the floating point numbers
        def format_significant(value, precision):
            return f"{value:.{precision}g}"

        # Saving losses to a CSV file
        with open(csv_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Writing headers
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'L2 Loss', 'L2 Std', 'L2 Max Outlier'])
            # Writing data with formatted values
            for i in range(epoch + 1):
                writer.writerow([
                    i + 1,
                    format_significant(train_losses[i], 4),
                    format_significant(val_losses[i], 4),
                    format_significant(l2_losses[i], 4),
                    format_significant(l2_stds[i], 4),
                    format_significant(l2_max_outlier[i], 4)
                ])

    def save_l2_histogram(self, epoch, l2_distances, n_bins=40):
        histogram_path = os.path.join(self.run_path, 'l2_histograms', f'validation_epoch_{epoch + 1}.png')
        # Calculate the histogram
        plt.hist(l2_distances, bins=n_bins, edgecolor='black')

        # Label the axes and title
        plt.xlabel('l2 distance')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of l2 distances epoch {epoch + 1}')

        # Save the histogram as a .png file
        plt.savefig(histogram_path)

        # Clear the current figure after saving it
        plt.clf()

    def save_l2_histogram_per_point(self, epoch, l2_per_point_dists, n_bins=20):
        # Number of points (rows in l2_per_point_dists)
        num_points = l2_per_point_dists.shape[0]
        # Prepare path for saving
        histogram_path = os.path.join(self.run_path, 'l2_histograms_per_point', f'validation_epoch_{epoch + 1}.png')

        # Create a larger figure to better accommodate multiple subplots
        fig, axs = plt.subplots(num_points, 1, figsize=(12, 4 * num_points))  # Height is now increased

        # Iterate over each row to plot the histogram
        for i in range(num_points):
            ax = axs[i]
            ax.hist(l2_per_point_dists[i], bins=n_bins, edgecolor='black')
            mean_val = np.mean(l2_per_point_dists[i])
            std_val = np.std(l2_per_point_dists[i])
            ax.set_title(f'Histogram for Point {i + 1} - Mean: {mean_val:.2f}, Std: {std_val:.2f}', fontsize=12)
            ax.set_xlabel('L2 distance in pixels', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)

        # Adjust layout to prevent overlap of subplots
        plt.tight_layout(pad=3.0)  # Increase padding to ensure titles and labels don't overlap
        # Save the figure
        plt.savefig(histogram_path)
        # Close the plot to free memory
        plt.close(fig)

    @staticmethod
    def find_points(confmaps):
        # points = find_peaks_soft_argmax(confmaps)
        points = tf_find_peaks_argmax(confmaps)
        return points

    def save_training_fig(self, epoch, train_losses, val_losses, start_fig_epoch=4):
        # Plot training and validation losses
        save_path = os.path.join(self.run_path, 'loss_graph.png')
        best_val_loss = np.min(val_losses)
        plt.figure(figsize=(10, 5))
        plt.title(f"Training and Validation Loss (Best Validation Loss: {best_val_loss:.7f})")
        plt.plot(np.arange(start_fig_epoch, epoch + 1), train_losses[start_fig_epoch:], label='Train')
        plt.plot(np.arange(start_fig_epoch, epoch + 1), val_losses[start_fig_epoch:], label='Val')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def save_configuration(self):
        with open(f"{self.run_path}/configuration.json", 'w') as file:
            json.dump(self.config, file, indent=4)

    def train_val_split(self, shuffle=True):
        """ Splits datasets into train and validation sets. """

        val_size = int(np.round(len(self.box) * self.val_fraction))
        idx = np.arange(len(self.box))
        if shuffle:
            np.random.shuffle(idx)

        val_idx = idx[:val_size]
        idx = idx[val_size:]

        return self.box[idx], self.confmaps[idx], self.box[val_idx], self.confmaps[val_idx], idx, val_idx

    def create_run_folders(self):
        run_path = os.path.join(self.base_output_path, self.run_name)
        if not self.clean:
            initial_run_path = run_path
            i = 1
            while os.path.exists(run_path):
                run_path = "%s_%02d" % (initial_run_path, i)
                i += 1
        if os.path.exists(run_path):
            shutil.rmtree(run_path)
        os.makedirs(run_path)
        os.makedirs(os.path.join(run_path, "weights"))
        os.makedirs(os.path.join(run_path, "histograms"))
        os.makedirs(os.path.join(run_path, "viz_pred"))
        os.makedirs(os.path.join(run_path, "l2_histograms"))
        os.makedirs(os.path.join(run_path, "l2_histograms_per_point"))
        print("Created folder:", run_path)
        code_dir_path = os.path.join(run_path, "training code")
        os.makedirs(code_dir_path)
        for file_name in os.listdir('.'):
            if file_name.endswith('.py'):
                full_file_name = os.path.join('.', file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, code_dir_path)
                    print(f"Copied {full_file_name} to {code_dir_path}")
        return run_path



if __name__ == '__main__':
    config_path = sys.argv[1]  # get the first argument
    print(config_path)
    trainer = Trainer(config_path)
    trainer.train()
