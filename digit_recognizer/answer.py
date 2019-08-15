from typing import Tuple, Collection, Optional

import numpy as np
import keras
from keras import layers

from digit_recognizer import TRAIN, TEST


class Param:
    def __init__(self, conv_filters: Collection[int], kernel_sizes: Collection[Tuple[int, int]],
                 strides: Collection[Tuple[int, int]], dense_units: Collection[int], noise_stddev: Optional[float]):
        assert len(conv_filters) == len(kernel_sizes) == len(strides)
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dense_units = dense_units
        self.noise_stddev = noise_stddev


def create_model(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], param: Param) -> keras.Model:
    inputs = keras.Input(input_shape)
    x = inputs
    if param is not None:
        x = layers.GaussianNoise(param.noise_stddev)(x)
    for filters, kernel_size, strides in zip(param.conv_filters, param.kernel_sizes, param.strides):
        x = layers.Conv2D(filters, kernel_size, strides=strides, activation='elu')(x)
    x = layers.Flatten()(x)
    for units in param.dense_units:
        x = layers.Dense(units, activation='elu')(x)
    outputs = layers.Dense(output_shape, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    train = np.loadtxt(TRAIN, delimiter=',', skiprows=1)
    x = train[:, 1:]
    y = train[:, 0]
    x = x.reshape((x.shape[0], 28, 28, 1)) / 255
    y = keras.utils.to_categorical(y)
    model = create_model(x.shape[1:], y.shape[-1],
                         Param(conv_filters=(32,) * 2, kernel_sizes=((3, 3),) * 2, strides=((1, 1),) * 2,
                               dense_units=(32,), noise_stddev=0.02))
    model.fit(x, y, batch_size=1024, epochs=20)  # , validation_split=0.2)

    test_x = np.loadtxt(TEST, delimiter=',', skiprows=1)
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1)) / 255
    test_y = model.predict(test_x, batch_size=1024)
    test_y = np.argmax(test_y, axis=1)
    with open('../submission.csv', 'x') as f:
        f.write('ImageId,Label\n')
        for i, label in enumerate(test_y, start=1):
            f.write(f'{i},{label}\n')


if __name__ == '__main__':
    main()
