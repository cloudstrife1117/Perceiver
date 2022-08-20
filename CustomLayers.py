"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from PositionEncodings import learnable_pos_embedding
import tensorflow as tf


class LatentArray(tf.keras.layers.Layer):
    def __init__(self, batch_size, latent_num, proj_dim):
        super(LatentArray, self).__init__()
        self.batch_size = batch_size
        self.latent_num = latent_num
        self.proj_dim = proj_dim
        self.latent_array = self.add_weight(
            shape=(self.latent_num, self.proj_dim), initializer="random_normal", trainable=True
        )
        self.pos_embedding = learnable_pos_embedding(pos_num=self.latent_num, proj_dim=self.proj_dim)

    def generate(self):
        latent_array_wpos = self.latent_array + self.pos_embedding
        latent_array_wpos = tf.broadcast_to(latent_array_wpos, [self.batch_size] + latent_array_wpos.shape)

        return latent_array_wpos
