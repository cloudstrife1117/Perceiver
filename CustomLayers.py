"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


class LatentArray(tf.keras.layers.Layer):
    def __init__(self, latent_num, proj_dim):
        super(LatentArray, self).__init__()
        self.latent_num = latent_num
        self.proj_dim = proj_dim
        self.latent_array = self.add_weight(
            shape=(self.latent_num, self.proj_dim), initializer="random_normal", trainable=True
        )

    def generate(self):
        return self.latent_array
