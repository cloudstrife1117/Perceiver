"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import layers
import math


def learnable_pos_embedding(pos_num, proj_dim):
    """
    Creates learnable position embeddings to all positions

    Returns
    -------
    position_embedding : tensor
      Position embeddings of all positions
    """
    positions = tf.range(start=0, limit=pos_num, delta=1)
    position_embedding_layer = layers.Embedding(
        input_dim=pos_num, output_dim=proj_dim
    )
    position_embedding = position_embedding_layer(positions)

    return position_embedding


def generate_coordinate_positions(input_space):
    # Get the dimension of the input space
    dim = len(input_space)
    # Generate the evenly space coordinate of each dimension within the range[-1, 1]
    dim_ranges = []
    for space in input_space:
        dim_ranges.append(tf.linspace(start=-1, stop=1, num=space))
    # Generate the even space positions given the coordinates of each dimension as output
    pos = tf.meshgrid(*dim_ranges)
    pos = tf.stack(pos, axis=dim)
    pos = tf.reshape(pos, [-1, dim])

    return pos


def generate_fourier_features(num_bands, input_space):
    # Generate coordinate positions
    pos = generate_coordinate_positions(input_space)
    # Generate Nyquist frequency(k evenly spaced frequencies between 1 and mu/2) for each dimension with k bands
    frequencies = tf.stack([tf.linspace(start=1., stop=space/2., num=num_bands) for space in input_space], axis=0)
    frequencies = tf.cast(frequencies, dtype=tf.float64)
    # Multiply the frequencies of each dimension with the according dimension of the position
    fourier_features = pos[:, :, None] * frequencies[None, :, :]
    fourier_features = tf.reshape(fourier_features, [-1, fourier_features.shape[1]*fourier_features.shape[2]])
    # Generate Pi Constant
    pi = tf.constant(math.pi, dtype=tf.float64)
    # Calculate the Sin and Cos Fourier Features
    fourier_features = tf.concat([tf.math.sin(tf.math.multiply(pi, fourier_features)),
                                  tf.math.cos(tf.math.multiply(pi, fourier_features))], axis=1)
    # Concat the original position with Fourier Features as output
    fourier_features = tf.concat([pos, fourier_features], axis=1)

    return fourier_features
