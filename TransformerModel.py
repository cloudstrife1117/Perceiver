"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from ImagePositionEmbedding import ImagePosEmbed
from CustomLayers import LatentArray, CrossAttentionTransformer, SelfAttentionTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


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

        # inputs are passed to this function as a placeholder for the latent array generation, just to construct the
        # functional model into a graph when it is called where the inputs isn't actually used
        latent_array_layer = LatentArray(batch_size=self.batch_size, latent_num=self.latent_num, proj_dim=self.proj_dim)
        latent_array = latent_array_layer(inputs)

        # Generating position encodings Fourier Features or learnable positions and combining with input
        embedding_layer = ImagePosEmbed(batch_size=self.batch_size, pos_num=inputs.shape[1]*inputs.shape[2], proj_dim=self.proj_dim, posEmbed=self.posEmbed)
        embeddings = embedding_layer(inputs)

        # Construct the initial CrossAttention Transformer
        ca_layer1 = CrossAttentionTransformer(proj_dim=self.proj_dim, num_heads=self.num_heads, dropout=self.dropout)
        latent1 = ca_layer1(latent_array, embeddings)

        # Construct the stacks of Latent Transformers
        sa_layer1 = SelfAttentionTransformer(proj_dim=self.proj_dim, num_heads=self.num_heads, dropout=self.dropout)
        sa_layer2 = SelfAttentionTransformer(proj_dim=self.proj_dim, num_heads=self.num_heads, dropout=self.dropout)
        latent1 = sa_layer1(latent1)
        latent1 = sa_layer2(latent1)

        # Average the output of final Latent Transformer with the number of latents(index dimension)
        mean_latent = tf.reduce_mean(latent1, axis=1)

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
        else:
            raise ValueError("Optimizer doesn't exists!")

        # Compile model
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

        # Train model and record training history
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, shuffle=True, validation_data=(X_val, y_val))
        self.history = history.history
