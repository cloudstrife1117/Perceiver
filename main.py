"""
@author: Jeng-Chung Lien
@email: masa67890@gmail.com
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from Modules.TransformerModel import TransformerModel
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def main():
    # Load Cifar-10 Dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    optimizer = "LAMB"
    init_learning_rate = 0.001
    end_learning_rate = 0.00000001  # 0.00000001
    loss_function = SparseCategoricalCrossentropy(from_logits=True)
    metrics = [SparseCategoricalAccuracy(name='Acc')]
    epochs = 50

    Perceiver = TransformerModel(input_shape=(32, 32, 3),
                                 batch_size=20,
                                 classes=10,
                                 latent_num=32,
                                 proj_dim=16,
                                 cross_num_heads=1,
                                 self_num_heads=8,
                                 block_num=1,
                                 stack_num=1,
                                 dropout=0.1,
                                 iter_num=1,
                                 model="Perceiver",
                                 posEmbed="FF",
                                 numbands=5,
                                 crop_size=(28, 28))

    Perceiver.summary()

    Perceiver.train(X_train=X_train,
                    X_val=X_test,
                    y_train=y_train,
                    y_val=y_test,
                    optimizer=optimizer,
                    init_lr=init_learning_rate,
                    end_lr=end_learning_rate,
                    loss=loss_function,
                    metrics=metrics,
                    epochs=epochs)


if __name__ == "__main__":
    main()
