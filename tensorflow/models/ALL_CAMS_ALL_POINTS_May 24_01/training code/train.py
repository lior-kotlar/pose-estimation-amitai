import json
import os
import shutil
import sys
from datetime import date
from time import time

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import Augmentor
import CallBacks
import Network
import preprocessor

import torch
try:
    torch.zeros(4).cuda()
except:
    print("No GPU found, doesnt use cuda")

print("TensorFlow version:", tf.__version__, flush=True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("TensorFlow is using GPU.", flush=True)
else:
    print("TensorFlow is using CPU.", flush=True)

print("Finished imports", flush=True)


class Trainer:
    def __init__(self, configuration_path):
        with open(configuration_path) as C:
            config = json.load(C)
            self.config = config
            self.batch_size = config['batch_size']
            self.num_epochs = config['epochs']
            self.batches_per_epoch = config['batches per epoch']
            self.val_fraction = config['val_fraction']
            self.base_output_path = config["base output path"]
            self.viz_idx = 1
            self.model_type = config["model type"]
            self.clean = bool(config["clean"])
            self.debug_mode = bool(config["debug mode"])
            self.preprocessor = preprocessor.Preprocessor(config)

        if self.debug_mode:
            self.batches_per_epoch = 1

        # Create the running folders
        self.run_name = f"{self.model_type}_{date.today().strftime('%b %d')}"
        self.run_path = self.create_run_folders()
        self.save_configuration()

        # Do preprocessing according to the model type
        self.preprocessor.do_preprocess()
        self.box, self.confmaps = self.preprocessor.get_box(), self.preprocessor.get_confmaps()

        # Get the right CNN architecture
        self.img_size = self.box.shape[1:]
        self.number_of_input_channels = self.box.shape[-1]
        self.num_output_channels = self.confmaps.shape[-1]
        self.network = Network.Network(config, image_size=self.img_size,
                                       number_of_output_channels=self.num_output_channels)
        self.model = self.network.get_model()

        # Split to train and validation
        self.train_box, self.train_confmap, self.val_box, self.val_confmap, _, _ = self.train_val_split()
        self.validation = (self.val_box, self.val_confmap)
        self.viz_sample = (self.val_box[self.viz_idx], self.val_confmap[self.viz_idx])
        print("img_size:", self.img_size)
        print("num_output_channels:", self.num_output_channels)

        # Create callback functions
        self.callbacker = CallBacks.CallBacks(config, self.run_path, self.model, self.viz_sample, self.validation)
        self.callbacks_list = self.callbacker.get_callbacks()
        self.history_callback = self.callbacker.get_history_callback()

        # Get augmentations generator
        self.augmentor = Augmentor.Augmentor(config, self.number_of_input_channels, self.num_output_channels)
        self.train_data_generator = self.augmentor.get_data_generator(self.box, self.confmaps)
        print("Creating generators - done!")

    def train(self):
        self.model.save(os.path.join(self.run_path, "initial_model.h5"))
        epoch0 = 0
        t0_train = time()
        training = self.model.fit(
            self.train_data_generator,
            initial_epoch=epoch0,
            epochs=self.num_epochs,
            verbose=1,
            steps_per_epoch=self.batches_per_epoch,
            shuffle=False,
            validation_data=self.validation,
            callbacks=self.callbacks_list,
            validation_steps=None
        )
        # Save final model
        self.model.history = self.history_callback.history
        self.model.save(os.path.join(self.run_path, "final_confmaps_model.h5"))
        elapsed_train = time() - t0_train
        print("Total runtime first loss: %.1f mins" % (elapsed_train / 60))

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
        """ Creates folders necessary for outputs of vision. """
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
        os.makedirs(os.path.join(run_path, "viz_pred"))
        os.makedirs(os.path.join(run_path, "histograms"))
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
    config_path = sys.argv[1]  # Get the first argument
    print(config_path)
    trainer = Trainer(config_path)
    trainer.train()
