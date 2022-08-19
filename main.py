"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import tensorflow as tf
from TransformerModel import TransformerModel


def main():
    # Load Cifar-10 Dataset
    # (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    Perceiver = TransformerModel(input_shape=(32, 32, 3),
                                 batch_size=20,
                                 classes=10,
                                 latent_num=32,
                                 proj_dim=16,
                                 num_heads=8,
                                 dropout=0.1,
                                 model="Perceiver",
                                 posEmbed="learnable")

    Perceiver.summary()


if __name__ == "__main__":
    main()
