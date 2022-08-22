"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import tensorflow as tf
from TransformerModel import TransformerModel
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def main():
    # Load Cifar-10 Dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    optimizer = "LAMB"
    learning_rate = 0.001
    loss_function = SparseCategoricalCrossentropy(from_logits=True)
    metrics = [SparseCategoricalAccuracy(name='Acc')]
    epochs = 10

    Perceiver = TransformerModel(input_shape=(32, 32, 3),
                                 batch_size=20,
                                 classes=10,
                                 latent_num=32,
                                 proj_dim=16,
                                 cross_num_heads=1,
                                 self_num_heads=8,
                                 stack_num=6,
                                 dropout=0.1,
                                 iter_num=2,
                                 model="Perceiver",
                                 posEmbed="FF")

    Perceiver.summary()

    Perceiver.train(X_train=X_train,
                    X_val=X_test,
                    y_train=y_train,
                    y_val=y_test,
                    optimizer=optimizer,
                    lr=learning_rate,
                    loss=loss_function,
                    metrics=metrics,
                    epochs=epochs)


if __name__ == "__main__":
    main()
