"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from ImagePositionEmbedding import ImagePosEmbed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LayerNormalization, Add, Dense, Dropout


class TransformerModel:
    def __init__(self, input_shape, batch_size, classes, latent_num, proj_dim, num_heads, dropout, model="Perceiver", posEmbed="FF"):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.classes = classes
        self.latent_num = latent_num
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.posEmbed = posEmbed
        self.history = None
        if model == "Perceiver":
            self.model = self.Perceiver()
        else:
            raise ValueError("Model doesn't exist!")

    def Perceiver(self):
        # Create Input layer
        inputs = Input(self.input_shape)

        embedding_layer = ImagePosEmbed(batch_size=self.batch_size, pos_num=inputs.shape[1]*inputs.shape[2], proj_dim=self.proj_dim, posEmbed=self.posEmbed)
        embeddings = embedding_layer(inputs)

        outputs = embeddings

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def summary(self):
        self.model.summary()
