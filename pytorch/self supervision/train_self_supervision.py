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
from tqdm import tqdm

# import wandb
np.random.seed(0)


class MyDataset(Dataset):
    def __init__(self, dataset_path, do_augmentations=False):
        self.dataset_path = dataset_path
        self.do_augmentations = do_augmentations
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.all_samples = os.listdir(self.dataset_path)
        self.do_horizontal_flip = True
        self.do_vertical_flip = True
        self.rotation_range = 180
        self.xy_shifts = 10
        self.scale_range = [0.8, 1.2]

    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, idx):
        file_name = self.all_samples[idx]
        path = os.path.join(self.dataset_path, file_name)
        which_mask = np.random.randint(0, 2)
        sample_box = np.load(path)[..., [0, 1, 2, 3 + which_mask]]
        sample_with_holes = self.create_holes(sample_box.copy())

        if self.do_augmentations:
            do_horizontal_flip = bool(np.random.randint(2)) and self.do_horizontal_flip
            do_vertical_flip = bool(np.random.randint(2)) and self.do_vertical_flip
            rotation_angle = np.random.randint(-self.rotation_range, self.rotation_range)
            shift_y_x = np.random.randint(-self.xy_shifts, self.xy_shifts, 2)
            scaling = np.random.uniform(self.scale_range[0], self.scale_range[1])
            sample_box = Augmentor.custom_augmentations(sample_box,
                                                        rotation_angle,
                                                        shift_y_x,
                                                        do_horizontal_flip,
                                                        do_vertical_flip,
                                                        scaling)
            sample_with_holes = Augmentor.custom_augmentations(sample_with_holes,
                                                               rotation_angle,
                                                               shift_y_x,
                                                               do_horizontal_flip,
                                                               do_vertical_flip,
                                                               scaling)

        # plt.imshow(sample_box[..., 1] + np.sum(sample_confmaps, axis=-1))
        # plt.show()
        sample_box, sample_with_holes = self.transform(sample_box).float(), self.transform(sample_with_holes).float()
        return sample_with_holes, sample_box

    @staticmethod
    def create_holes(image, hole_wing=8, hole_body=16):
        # body = (np.sum(image[..., :-1], axis=-1) > 2).astype(float)
        mask = image[..., -1]

        hole_wing = int(np.sqrt(np.count_nonzero(mask > 0)) // 2)
        # hole_body = int(np.sqrt(np.count_nonzero(body > 0)) // 10)
        mask_coords = np.argwhere(mask > 0)
        image_coords = np.argwhere(np.sum(image[..., [0, 1, 2]], axis=-1) > 0)

        mask_holes = mask_coords[np.random.choice(mask_coords.shape[0], 3, replace=False)]
        image_holes = image_coords[np.random.choice(image_coords.shape[0], 5, replace=False)]

        for x, y in mask_holes:
            x = max(0, x - hole_wing // 2)
            y = max(0, y - hole_wing // 2)
            image[x:x + hole_wing, y:y + hole_wing, :] = 0
        for x, y in image_holes:
            x = max(0, x - hole_body // 2)
            y = max(0, y - hole_body // 2)
            image[x:x + hole_body, y:y + hole_body, :] = 0

        # plt.imshow(image[..., [1, 1, -1]])
        # plt.show()

        return image


class Trainer:
    def __init__(self, dataset_path, train_config_path="train_self_supervision_config.json"):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(self.dataset_path, 'train')
        self.validation_path = os.path.join(self.dataset_path, 'validation')
        self.batch_size = 10
        self.num_epochs = 5000
        self.batches_per_epoch = -1
        self.val_fraction = 0.1
        self.base_output_path = ""
        self.viz_idx = 1
        self.clean = False
        with open(train_config_path) as C:
            config = json.load(C)
            self.config = config

        # create the running folders
        self.run_name = f"self_supervision_{date.today().strftime('%b %d')}"
        self.run_path = self.create_run_folders()
        self.save_configuration()

        # get the right model architecture
        self.network = Network.Network(config, image_size=(192, 192, 4),
                                       num_output_channels=4)
        self.model = self.network.get_model()

        # Create your datasets
        self.train_dataset = MyDataset(self.train_path, do_augmentations=True)
        self.val_dataset = MyDataset(self.validation_path, do_augmentations=False)
        # Create your dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           drop_last=True, )
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=10)

    def train(self):
        t0_train = time()
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device", self.device)
        model = self.model.to(self.device)
        num_epochs = self.num_epochs
        best_loss = float('inf')
        # For plotting loss graph
        train_losses = []
        val_losses = []
        # Define the criterion and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # you can adjust the learning rate

        torch.manual_seed(0)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            running_loss = 0.0

            # Training loop
            model.train()
            for i, (inputs, targets) in enumerate(self.train_dataloader):
                batch_size = inputs.size(0)
                if i % 10 == 0:
                    print(f"batch number is {i + 1}, batch size is {batch_size}")
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                outputs = model(inputs)

                # l1_lambda = 0.00001  # The regularization hyperparameter
                # l1_norm = sum(p.abs().sum() for p in model.parameters())

                # loss = criterion(outputs, targets) + l1_lambda * l1_norm
                loss = criterion(outputs, targets)

                loss.backward()

                # Clip gradients to avoid exploding gradient problem
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_grad_norm as needed

                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # Update the progress bar
            epoch_loss = running_loss / len(self.train_dataloader)  # Normalize the loss by the total number of samples

            print(f'Train Loss: {epoch_loss:.4f}')
            train_losses.append(epoch_loss)

            # Validation phase
            model.eval()
            val_running_loss = 0.0

            # validation loop
            for i, (inputs, targets) in enumerate(self.val_dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # forward
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                # statistics
                val_running_loss += loss.item() * inputs.size(0)

                # Save the validation image before and after going through the network
                if i == 0:  # Save for the first batch
                    # Convert tensors to numpy arrays
                    self.save_validation_image(epoch, inputs, outputs)

            epoch_loss = val_running_loss / len(self.val_dataloader)
            print(f'Val Loss: {epoch_loss:.4f}')

            # deep copy the model (for saving the best model)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model_path = os.path.join(self.run_path, 'best_model.pth')
                torch.save(best_model_wts, save_model_path)

            # Save losses for plotting
            val_losses.append(epoch_loss)

            # functions to preform every end of epoch
            self.save_checkpoint(epoch, epoch_loss, model, optimizer)
            self.save_training_fig(epoch, train_losses, val_losses)
        elapsed_train = time() - t0_train
        print("Total runtime first loss: %.1f mins" % (elapsed_train / 60))

    def save_validation_image(self, epoch, inputs, outputs):
        inputs_np = inputs.detach().cpu().numpy()[0]
        outputs_np = outputs.detach().cpu().numpy()[0]
        inputs_np = np.transpose(inputs_np, (1, 2, 0))
        outputs_np = np.transpose(outputs_np, (1, 2, 0))
        inputs_np = np.clip(inputs_np, 0, 1)
        outputs_np = np.clip(outputs_np, 0, 1)
        # Save the numpy arrays
        np.save(os.path.join(self.run_path, f'inputs_epoch{epoch}.npy'), inputs_np)
        np.save(os.path.join(self.run_path, f'outputs_epoch{epoch}.npy'), outputs_np)
        # Convert the numpy arrays to images and save
        # Select the channels [1, 1, -1] for both inputs and outputs
        plt.imsave(os.path.join(self.run_path, f'inputs_epoch{epoch}.png'), inputs_np[..., [1, 1, -1]])
        plt.imsave(os.path.join(self.run_path, f'outputs_epoch{epoch}.png'), outputs_np[..., [1, 1, -1]])

    def display_batch(self, inputs):
        image = inputs[0, ...]
        image = image.detach().cpu().numpy()
        image = np.transpose(image, [1, 2, 0])
        plt.imshow(image[..., [0, 1, 2]])
        plt.show()

    def save_checkpoint(self, epoch, epoch_loss, model, optimizer):
        save_path = os.path.join(self.run_path, 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, save_path)

    def save_training_fig(self, epoch, train_losses, val_losses):
        # Plot training and validation losses
        save_path = os.path.join(self.run_path, 'loss_graph.png')
        best_val_loss = np.min(val_losses)
        plt.figure(figsize=(10, 5))
        plt.title(f"Training and Validation Loss (Best Validation Loss: {best_val_loss:.4f})")
        plt.plot(np.arange(1, epoch + 2), train_losses, label='Train')
        plt.plot(np.arange(1, epoch + 2), val_losses, label='Val')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def save_configuration(self):
        with open(f"{self.run_path}/configuration.json", 'w') as file:
            json.dump(self.config, file, indent=4)

    def create_run_folders(self):
        """ Creates subfolders necessary for outputs of vision. """
        run_path = os.path.join(self.base_output_path, self.run_name)

        if not self.clean:
            initial_run_path = run_path
            i = 1
            while os.path.exists(run_path):  # and not is_empty_run(run_path):
                run_path = "%s_%02d" % (initial_run_path, i)
                i += 1

        if os.path.exists(run_path):
            shutil.rmtree(run_path)

        os.makedirs(run_path)
        os.makedirs(os.path.join(run_path, "weights"))
        os.makedirs(os.path.join(run_path, "viz_pred"))
        os.makedirs(os.path.join(run_path, "viz_confmaps"))
        print("Created folder:", run_path)

        return run_path


if __name__ == '__main__':
    # data_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\as_numpy_arrays"
    # trainer = Trainer(data_path)
    # trainer.train()

    data_path = "../self supervision dataset"
    trainer = Trainer(data_path)
    trainer.train()
