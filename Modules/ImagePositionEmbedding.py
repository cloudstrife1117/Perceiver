"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from .PositionEncodings import learnable_pos_embedding, generate_fourier_features
import tensorflow as tf
from tensorflow.keras import layers


class ImagePosEmbed(layers.Layer):
    def __init__(self, batch_size, pos_num, proj_dim, posEmbed="FF", num_bands=15):
        super(ImagePosEmbed, self).__init__()
        self.batch_size = batch_size
        self.proj_dim = proj_dim
        self.posEmbed = posEmbed
        self.img_embedding_layer = layers.Dense(units=proj_dim)
        if posEmbed == "learnable":
            self.pos_embedding = learnable_pos_embedding(pos_num=pos_num, proj_dim=self.proj_dim)
        elif posEmbed == "FF":
            self.num_bands = num_bands

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        x = x / 255.  # Normalize Image Values
        input_space = (x.shape[1], x.shape[2])
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]])

        if self.posEmbed == "FF":
            fourier_features = generate_fourier_features(num_bands=self.num_bands, input_space=input_space)
            fourier_features = tf.broadcast_to(fourier_features, [self.batch_size] + fourier_features.shape)
            x = tf.concat([x, fourier_features], axis=-1)
        elif self.posEmbed == "learnable":
            x = self.img_embedding_layer(x)
            x = x + self.pos_embedding
        else:
            raise ValueError("Position Embedding Method doesn't exist!")

        return x
