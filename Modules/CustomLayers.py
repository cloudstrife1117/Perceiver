"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from .PositionEncodings import learnable_pos_embedding
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Add, Dense, Dropout
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow_addons.activations import gelu


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

    def call(self, inputs):
        latent_array_wpos = self.latent_array + self.pos_embedding
        latent_array_wpos = tf.broadcast_to(latent_array_wpos, [self.batch_size] + latent_array_wpos.shape)

        return latent_array_wpos


class MLP(tf.keras.layers.Layer):
    def __init__(self, proj_dim, dropout):
        super(MLP, self).__init__()
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.dense1 = Dense(self.proj_dim * 1, activation=gelu)
        self.dense2 = Dense(self.proj_dim)

    def call(self, inputs):
        x1 = self.dense1(inputs)
        x1 = Dropout(self.dropout)(x1)
        x2 = self.dense2(x1)
        x2 = Dropout(self.dropout)(x2)

        return x2


class CrossAttentionTransformer(tf.keras.layers.Layer):
    def __init__(self, proj_dim, num_heads, dropout):
        super(CrossAttentionTransformer, self).__init__()
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha_layer = MultiHeadAttention(head_size=self.proj_dim, num_heads=self.num_heads, output_size=self.proj_dim, dropout=self.dropout)
        self.mlp_layer = MLP(proj_dim=self.proj_dim, dropout=self.dropout)
        self.q_norm_layer1 = LayerNormalization(epsilon=1e-6)
        self.kv_norm_layer1 = LayerNormalization(epsilon=1e-6)
        self.q_norm_layer2 = LayerNormalization(epsilon=1e-6)

    def call(self, qkv_input):
        q_input = qkv_input[0]
        kv_input = qkv_input[1]
        q_norm1 = self.q_norm_layer1(q_input)
        kv_norm1 = self.kv_norm_layer1(kv_input)
        q_mha = self.mha_layer([q_norm1, kv_norm1, kv_norm1])
        q_add1 = Add()([q_mha, q_input])
        q_norm2 = self.q_norm_layer2(q_add1)
        q_mlp1 = self.mlp_layer(q_norm2)
        q_add2 = Add()([q_mlp1, q_add1])

        return q_add2


class SelfAttentionTransformer(tf.keras.layers.Layer):
    def __init__(self, proj_dim, num_heads, dropout):
        super(SelfAttentionTransformer, self).__init__()
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha_layer = MultiHeadAttention(head_size=self.proj_dim, num_heads=self.num_heads, dropout=self.dropout)
        self.mlp_layer = MLP(proj_dim=self.proj_dim, dropout=self.dropout)
        self.qvk_norm_layer1 = LayerNormalization(epsilon=1e-6)
        self.qvk_norm_layer2 = LayerNormalization(epsilon=1e-6)

    def call(self, qkv_input):
        qkv_norm = self.qvk_norm_layer1(qkv_input)
        qkv_mha = self.mha_layer([qkv_norm, qkv_norm, qkv_norm])
        qkv_add1 = Add()([qkv_mha, qkv_input])
        qkv_norm2 = self.qvk_norm_layer2(qkv_add1)
        qkv_mlp1 = self.mlp_layer(qkv_norm2)
        qkv_add2 = Add()([qkv_mlp1, qkv_add1])

        return qkv_add2


class LatentTransformer(tf.keras.layers.Layer):
    def __init__(self, proj_dim, num_heads, dropout, stack_num):
        super(LatentTransformer, self).__init__()
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.stack_num = stack_num
        self.sa_layers = [SelfAttentionTransformer(proj_dim=self.proj_dim,
                                                   num_heads=self.num_heads,
                                                   dropout=self.dropout) for _ in range(self.stack_num)]

    def call(self, qkv_input):
        for i in range(self.stack_num):
            qkv_input = self.sa_layers[i](qkv_input)

        return qkv_input
