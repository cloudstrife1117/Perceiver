"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from .ImagePositionEmbedding import ImagePosEmbed
from .CustomLayers import LatentArray, CrossAttentionTransformer, LatentTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import LAMB


class TransformerModel:
    def __init__(self, input_shape, batch_size, classes, latent_num, proj_dim, cross_num_heads, self_num_heads, block_num, stack_num, dropout, iter_num, model="Perceiver", posEmbed="FF"):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.classes = classes
        self.latent_num = latent_num
        self.proj_dim = proj_dim
        self.cross_num_heads = cross_num_heads
        self.self_num_heads = self_num_heads
        self.block_num = block_num
        self.stack_num = stack_num
        self.dropout = dropout
        self.iter_num = iter_num
        self.posEmbed = posEmbed
        self.history = None
        if model == "Perceiver":
            self.model = self.Perceiver()
        else:
            raise ValueError("Model doesn't exist!")

    def Perceiver(self):
        # Create Input layer
        inputs = Input(self.input_shape)

        # inputs are passed to this function as a placeholder for the latent array generation, just to construct the
        # functional model into a graph when it is called where the inputs isn't actually used
        latent_array_layer = LatentArray(batch_size=self.batch_size, latent_num=self.latent_num, proj_dim=self.proj_dim)
        latent_array = latent_array_layer(inputs)

        # Generating position encodings Fourier Features or learnable positions and combining with input
        embedding_layer = ImagePosEmbed(batch_size=self.batch_size, pos_num=inputs.shape[1]*inputs.shape[2], proj_dim=self.proj_dim, posEmbed=self.posEmbed)
        embeddings = embedding_layer(inputs)

        # Construct the initial CrossAttention Transformer
        ca_layer1 = CrossAttentionTransformer(proj_dim=self.proj_dim, num_heads=self.cross_num_heads, dropout=self.dropout)

        # Construct following CrossAttention Transformer to repeat exclude first initial CrossAttention when repeat on after first iteration
        ca_layers = [CrossAttentionTransformer(proj_dim=self.proj_dim, num_heads=self.cross_num_heads, dropout=self.dropout) for _ in range(self.block_num)]

        # Construct the multiple stacks of Latent Transformers following each CrossAttention
        latent_transformers = [LatentTransformer(proj_dim=self.proj_dim, num_heads=self.self_num_heads, dropout=self.dropout, stack_num=self.stack_num) for _ in range(self.block_num)]

        # Pass trough first initial CrossAttention Transformer layer and the first Latent Transformer
        latent = ca_layer1(latent_array, embeddings)
        latent = latent_transformers[0](latent)
        # Pass trough the second and following CrossAttention Transformer layer with Latent Transformer each after CrossAttention
        for i in range(1, self.block_num):
            latent = ca_layers[i](latent, embeddings)
            latent = latent_transformers[i](latent)

        # Repeat to unroll depth for iter_num-1 times iteration following after first iteration(shared weights), exclude first inital CrossAttention
        for _ in range(self.iter_num-1):
            for i in range(self.block_num):
                latent = ca_layers[i](latent, embeddings)
                latent = latent_transformers[i](latent)

        # Average the output of final Latent Transformer with the number of latents(index dimension)
        mean_latent = tf.reduce_mean(latent, axis=1)

        # Pass the averaged latent trough a linear layer to output the number of classes of logits
        logits = Dense(units=self.classes)(mean_latent)

        model = Model(inputs=inputs, outputs=logits)

        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, X_val, y_train, y_val, optimizer, lr, loss, metrics, epochs):
        # Apply optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)
        elif optimizer == 'LAMB':
            opt = LAMB(learning_rate=lr)
        else:
            raise ValueError("Optimizer doesn't exists!")

        # Compile model
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

        # Train model and record training history
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, shuffle=True, validation_data=(X_val, y_val))
        self.history = history.history
