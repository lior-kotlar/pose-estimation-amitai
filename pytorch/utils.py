import tensorflow as tf
import torch
import numpy as np


def tf_find_peaks_argmax(x):
    """ Finds the maximum value in each channel and returns the location and value.
    Args:
        x: rank-4 tensor (samples, height, width, channels)

    Returns:
        peaks: rank-3 tensor (samples, [x, y, val], channels)
    """

    # Store input shape
    in_shape = tf.shape(x)

    image_size = int(in_shape[1])

    # Flatten height/width dims
    flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])

    # Find peaks in linear indices
    idx = tf.argmax(flattened, axis=1)

    # Convert linear indices to subscripts
    rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
    cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])

    # Dumb way to get actual values without indexing
    vals = tf.math.reduce_max(flattened, axis=1)
    vals = tf.cast(vals, tf.float32)
    # Return N x 3 x C tensor
    pred = tf.stack([
        tf.cast(cols, tf.float32),
        tf.cast(rows, tf.float32),
        vals
    ], axis=1)

    pred = np.transpose(pred, (0, 2, 1))
    pred = pred[..., :2]
    # pred = pred / image_size  # normalize points

    return pred


def find_peaks_soft_argmax(x):

    heatmap = torch.from_numpy(x).float()

    # Adjust the dimensions to [batch_size, num_channels, height, width]
    heatmap = heatmap.permute(0, 3, 1, 2)

    batch_size, num_channels, height, width = heatmap.shape

    # Create normalized grids for x and y coordinates
    y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, steps=height),
                                    torch.linspace(0, 1, steps=width))
    y_grid, x_grid = y_grid.to(heatmap.device), x_grid.to(heatmap.device)

    # Compute the weighted sums for x and y coordinates across all images and channels
    weighted_sum_x = (x_grid * heatmap).sum(dim=[2, 3])
    weighted_sum_y = (y_grid * heatmap).sum(dim=[2, 3])

    # Compute the sum of all weights (pixel values) for normalization
    total_weight = heatmap.sum(dim=[2, 3])

    # Calculate the centroid coordinates
    centroid_x = weighted_sum_x / total_weight
    centroid_y = weighted_sum_y / total_weight

    # Convert normalized coordinates to image dimensions
    centroid_x = centroid_x * (width - 1)
    centroid_y = centroid_y * (height - 1)

    # Clamp centroid coordinates to ensure they are within image boundaries
    centroid_x = torch.clamp(centroid_x, 0, width - 1)
    centroid_y = torch.clamp(centroid_y, 0, height - 1)

    # Combine the coordinates
    centroids = torch.stack([centroid_x, centroid_y], dim=-1)

    return np.array(centroids)


