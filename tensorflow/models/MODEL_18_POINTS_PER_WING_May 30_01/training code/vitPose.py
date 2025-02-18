import tensorflow as tf
from tensorflow.keras import layers


# Define a custom layer for patch extraction
class PatchExtractionLayer(layers.Layer):
    def __init__(self, patch_size, num_patches, num_input_channels, **kwargs):
        super(PatchExtractionLayer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_input_channels = num_input_channels

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, shape=(-1, self.num_patches, self.patch_size * self.patch_size * self.num_input_channels))
        return patches

    def get_config(self):
        config = super(PatchExtractionLayer, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config

# Define a custom layer for reshaping embeddings
class ReshapeEmbeddingsLayer(layers.Layer):
    def __init__(self, image_size, patch_size, **kwargs):
        super(ReshapeEmbeddingsLayer, self).__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size

    def call(self, embeddings):
        return tf.reshape(embeddings, shape=(-1, self.image_size // self.patch_size, self.image_size // self.patch_size, embeddings.shape[-1]))

    def get_config(self):
        config = super(ReshapeEmbeddingsLayer, self).get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        })
        return config

# Define a function to embed patches using a linear projection
def patch_embedding(patches, projection_dim):
    patch_embedding = layers.Dense(units=projection_dim)(patches)
    return patch_embedding

# Define a function to add positional embeddings to patch embeddings
def positional_embedding(embeddings, num_patches, projection_dim):
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    position_embedding = tf.expand_dims(position_embedding, axis=0)
    embeddings = embeddings + position_embedding
    return embeddings

# Define a function to create a transformer layer
def transformer_layer(x, num_heads, projection_dim, fully_connected_expand):
    # Create a multi-head attention layer
    attention_output = layers.MultiHeadAttention(num_heads=num_heads,
                                                 key_dim=projection_dim,
                                                 dropout=0.1)(x, x)
    # Add skip connection
    x = layers.Add()([x, attention_output])
    # Layer normalization
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Create a feed-forward layer
    ffn_output = layers.Dense(units=projection_dim * fully_connected_expand, activation="relu")(x)
    ffn_output = layers.Dense(units=projection_dim)(ffn_output)
    # Add skip connection
    x = layers.Add()([x, ffn_output])
    # Layer normalization
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# Define a function to reconstruct an image from patch embeddings
def image_reconstruction(embeddings, image_size, num_output_channels, patch_size):
    # Reshape the embeddings to the shape of patches
    patches = ReshapeEmbeddingsLayer(image_size, patch_size)(embeddings)
    decoder = patches
    for _ in range(3):
        decoder = layers.Conv2DTranspose(decoder.shape[-1] // 2,
                                  kernel_size=3, strides=2, padding="same",
                                  activation=layers.LeakyReLU(alpha=0.1),
                                  kernel_initializer="glorot_normal")(decoder)

    decoder = layers.Conv2DTranspose(num_output_channels,
                              kernel_size=3, strides=2, padding="same",
                              activation=layers.LeakyReLU(alpha=0.1),
                              kernel_initializer="glorot_normal")(decoder)
    return decoder


# Define a function to create a vision transformer model
def vision_transformer(image_size=192,
                        patch_size=16,
                        projection_dim=256,
                        num_heads =8,
                        transformer_layers=8 ,
                        num_input_channels=4,
                        num_output_channels=10,
                        fully_connected_expand=4):

    num_patches = (image_size // patch_size) ** 2
    # Input layer
    inputs = layers.Input(shape=(image_size, image_size, num_input_channels))
    # Create patches
    patches = PatchExtractionLayer(patch_size, num_patches, num_input_channels)(inputs)
    # Embed patches
    x = patch_embedding(patches, projection_dim)
    # Add positional embeddings
    x = positional_embedding(x, num_patches, projection_dim)
    # Create N transformer layers
    for _ in range(transformer_layers):
        x = transformer_layer(x, num_heads, projection_dim, fully_connected_expand)
    # Reconstruct the image from the embeddings
    outputs = image_reconstruction(x, image_size, num_output_channels, patch_size)
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError())
    # Print the model summary
    model.summary()
    return model

if __name__ == '__main__':
    # Create the model
    model_vit = vision_transformer()
