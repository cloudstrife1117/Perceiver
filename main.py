import tensorflow as tf
import math


def generate_positions(input_space):
    # Get the dimension of the input space
    dim = len(input_space)
    # Generate the evenly space coordinate of each dimension within the range[-1, 1]
    dim_ranges = []
    for space in input_space:
        dim_ranges.append(tf.linspace(start=-1, stop=1, num=space))
    # Generate the even space positions given the coordinates of each dimension as output
    pos = tf.meshgrid(*dim_ranges)
    pos = tf.stack(pos, axis=dim)
    pos = tf.reshape(pos, [-1, dim])

    return pos


def generate_fourier_features(pos, num_bands, input_space):
    # Generate Nyquist frequency(k evenly spaced frequencies between 1 and mu/2) for each dimension with k bands
    frequencies = tf.stack([tf.linspace(start=1., stop=space/2., num=num_bands) for space in input_space], axis=0)
    frequencies = tf.cast(frequencies, dtype=tf.float64)
    # Multiply the frequencies of each dimension with the according dimension of the position
    fourier_features = pos[:, :, None] * frequencies[None, :, :]
    fourier_features = tf.reshape(fourier_features, [-1, fourier_features.shape[1]*fourier_features.shape[2]])
    # Generate Pi Constant
    pi = tf.constant(math.pi, dtype=tf.float64)
    # Calculate the Sin and Cos Fourier Features
    fourier_features = tf.concat([tf.math.sin(tf.constant(pi * fourier_features)),
                                  tf.math.cos(tf.constant(pi * fourier_features))], axis=1)
    # Concat the original position with Fourier Features as output
    fourier_features = tf.concat([pos, fourier_features], axis=1)

    return fourier_features


def main():
    # Example to add a 2D Fourier Features to one of the CIFAR-10 image data
    input_space = (32, 32)
    pos = generate_positions(input_space)
    fourier_features = generate_fourier_features(pos, num_bands=15, input_space=input_space)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    norm_image = X_train[1] / 255.
    image_withFF = tf.concat([tf.reshape(norm_image, [-1, 3]), fourier_features], axis=1)
    print(image_withFF)


if __name__ == "__main__":
    main()