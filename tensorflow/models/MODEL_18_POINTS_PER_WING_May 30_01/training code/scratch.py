import tensorflow as tf
from tensorflow.keras import layers, models, Model


class VITPoseDecoder(Model):
    def __init__(self, config, num_patches, num_output_channels, **kwargs):
        super(VITPoseDecoder, self).__init__(**kwargs)
        self.projection_dim = config["projection dim"]
        self.num_heads = config["num heads"]
        self.transformer_layers = config["transformer layers"]
        self.fully_connected_expand = config["fully connected expand"]
        self.num_patches = num_patches
        self.num_output_channels = num_output_channels

        self.patch_embedding = layers.Dense(self.projection_dim)
        self.position_embedding = layers.Embedding(input_dim=self.num_patches, output_dim=self.projection_dim)
        self.transformer_blocks = [self.create_transformer_layer() for _ in range(self.transformer_layers)]
        self.output_layer = layers.Conv2DTranspose(num_output_channels, kernel_size=3, strides=2, padding="same",
                                                   activation='sigmoid')

    def create_transformer_layer(self):
        return models.Sequential([
            layers.MultiHeadAttention(self.num_heads, key_dim=self.projection_dim, dropout=0.1),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(self.projection_dim * self.fully_connected_expand, activation='relu'),
            layers.Dense(self.projection_dim),
            layers.LayerNormalization(epsilon=1e-6),
        ])

    def call(self, inputs):
        x = self.patch_embedding(inputs)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        x += self.position_embedding(positions)
        for block in self.transformer_blocks:
            x = block(x)
        # Reshape to fit Conv2DTranspose input requirements
        x = tf.reshape(x, shape=(-1, int(self.num_patches ** 0.5), int(self.num_patches ** 0.5), self.projection_dim))
        x = self.output_layer(x)
        return x


class MultiCamVITPose(Model):
    def __init__(self, config, image_size, num_output_channels, **kwargs):
        super(MultiCamVITPose, self).__init__(**kwargs)
        self.image_size = image_size
        self.num_output_channels = num_output_channels
        self.patch_size = config["patch size"]
        self.num_patches = (image_size[0] // self.patch_size) ** 2

        # Shared Encoder
        self.shared_encoder = self.create_transformer_encoder(config, image_size)

        # Shared Decoder
        self.shared_decoder = VITPoseDecoder(config, self.num_patches, num_output_channels)

    def create_transformer_encoder(self, config, image_size):
        # Simplified encoder for brevity
        return models.Sequential([
            layers.Dense(config["projection dim"], activation='relu'),
            layers.LayerNormalization(epsilon=1e-6)
        ])

    def call(self, inputs):
        # Process inputs from multiple cameras
        cams = [inputs[..., i * 4:(i + 1) * 4] for i in range(4)]  # Assuming 4 cameras

        encoded_cams = [self.shared_encoder(cam) for cam in cams]

        # Merge features from all cameras
        merged_features = layers.Concatenate()(encoded_cams)
        decoded_output = self.shared_decoder(merged_features)

        return decoded_output


if __name__ == "__main__":
    import numpy as np
    # Example configuration and model instantiation
    config = {
        "patch size": 16,
        "projection dim": 256,
        "num heads": 8,
        "transformer layers": 8,
        "fully connected expand": 4,
    }

    image_size = (192, 192, 16)  # Example image size with 16 channels to accommodate 4 cameras
    num_output_channels = 40  # Example output channels for pose estimation

    model = MultiCamVITPose(config, image_size, num_output_channels)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Model summary
    model.build(input_shape=(None, *image_size))
    model.summary()
    x = np.random.random((2, 192, 192, 16))
    y = model(x)
