from typing import Tuple, Collection, Optional
import os

import numpy as np
import keras
from keras import layers
from keras import backend as K

from digit_recognizer import TRAIN, TEST, RESULTS


def center_loss(y_true, y_pred, margin: float):
    y_true = K.cast(y_true, y_pred.dtype)
    return K.categorical_crossentropy(y_true, y_pred - y_true * margin, from_logits=True)


class CenterLoss:
    def __init__(self, margin: float):
        self.margin = margin
        self.__name__ = 'CenterLoss'

    def __call__(self, y_true, y_pred):
        return center_loss(y_true, y_pred, self.margin)


class Param:
    def __init__(self, conv_filters: Collection[int], kernel_sizes: Collection[Tuple[int, int]],
                 strides: Collection[Tuple[int, int]],
                 pool_sizes: Collection[Tuple[int, int]], pool_strides: Collection[Tuple[int, int]],
                 dense_units: Collection[int], use_l2_constrained: bool, l2_scale: Optional[float],
                 noise_stddev: Optional[float]):
        assert len(conv_filters) == len(kernel_sizes) == len(strides) == len(pool_sizes) == len(pool_strides)
        for pool_size, pool_stride in zip(pool_sizes, pool_strides):
            is_size_none, is_stride_none = pool_size is None, pool_stride is None
            assert not (is_size_none ^ is_stride_none)
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.dense_units = dense_units
        self.use_l2_constrained = use_l2_constrained
        self.l2_scale = l2_scale
        self.noise_stddev = noise_stddev


def create_model(input_shape: Tuple[int, ...], output_shape: int, param: Param) -> keras.Model:
    inputs = keras.Input(input_shape)
    x = inputs
    if param.noise_stddev is not None:
        x = layers.GaussianNoise(param.noise_stddev)(x)
    x = layers.Lambda(lambda z: z - K.mean(z, axis=1, keepdims=True))(x)
    # x = layers.Lambda(lambda z: z / K.sqrt(K.var(z, axis=1, keepdims=True)))(x)
    for filters, kernel_size, strides, pool_size, pool_stride \
            in zip(param.conv_filters, param.kernel_sizes, param.strides, param.pool_sizes, param.pool_strides):
        x = layers.Conv2D(filters, kernel_size, strides=strides)(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.ELU()(x)
        if pool_size is not None:
            x = layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride)(x)
    x = layers.Flatten()(x)
    for units in param.dense_units:
        x = layers.Dense(units, activation='elu')(x)
    if param.use_l2_constrained:
        x = layers.Lambda(lambda z: K.l2_normalize(z, axis=1) * param.l2_scale)(x)
        outputs = layers.Dense(output_shape, kernel_constraint=keras.constraints.UnitNorm(),
                               use_bias=False, activation='softmax')(x)
    else:
        outputs = layers.Dense(output_shape, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def complexize_kernel(layer, units, *, activation=None, **kwargs):
    """乗算的なKerasレイヤーをFunctional APIとして複素化する。"""

    def complexized(x_re, x_im):
        layer_re = layer(units, **kwargs)
        layer_im = layer(units, **kwargs)
        x_re_tmp = layers.Subtract()([layer_re(x_re), layer_im(x_im)])
        x_im_tmp = layers.Add()([layer_re(x_im), layer_im(x_re)])
        x_re, x_im = x_re_tmp, x_im_tmp
        if activation is None:
            return x_re, x_im
        x_abs = layers.Lambda(lambda x_cpx: K.sqrt(K.square(x_cpx[0]) + K.square(x_cpx[1])))([x_re, x_im])
        x_act = activation(x_abs)
        projection = layers.Lambda(lambda a: a[0] * a[1] / a[2])
        x_re = projection([x_act, x_re, x_abs])
        x_im = projection([x_act, x_im, x_abs])
        return x_re, x_im

    return complexized


def create_complex_model(input_shape: Tuple[int, ...], output_shape: int, param: Param) -> keras.Model:
    inputs = keras.Input(input_shape)
    x = inputs
    if param is not None:
        x = layers.GaussianNoise(param.noise_stddev)(x)
    x_re = x
    x_im = layers.Lambda(lambda z: K.zeros(((1,) + input_shape)))([])

    for filters, kernel_size, strides, pool_size, pool_stride \
            in zip(param.conv_filters, param.kernel_sizes, param.strides, param.pool_sizes, param.pool_strides):
        x_re, x_im = complexize_kernel(layers.Conv2D, filters, kernel_size=kernel_size,
                                       strides=strides, activation=layers.Activation("tanh"))(x_re, x_im)
        x_re = layers.BatchNormalization(axis=-1)(x_re)
        x_im = layers.BatchNormalization(axis=-1)(x_im)
        if pool_size is not None:
            x_re = layers.AveragePooling2D(pool_size=pool_size, strides=pool_stride)(x_re)
            x_im = layers.AveragePooling2D(pool_size=pool_size, strides=pool_stride)(x_im)
    x_re = layers.Flatten()(x_re)
    x_im = layers.Flatten()(x_im)
    for units in param.dense_units:
        x_re, x_im = complexize_kernel(layers.Dense, units, activation=layers.Activation("tanh"))(x_re, x_im)
    # x = layers.Lambda(lambda d: K.sqrt(K.square(d[0]) + K.square(d[1])))([x_re, x_im])
    if param.use_l2_constrained:
        x = layers.Lambda(lambda z: K.l2_normalize(z, axis=1) * param.l2_scale)(x_re)
        outputs = layers.Dense(output_shape, kernel_constraint=keras.constraints.UnitNorm(),
                               use_bias=False, activation='softmax')(x)
    else:
        outputs = layers.Dense(output_shape, activation='softmax')(x_re)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    param = Param(conv_filters=(16, 32, 64, 128), kernel_sizes=((3, 3),) * 4, strides=((1, 1),) * 4,
                  pool_sizes=((3, 3), None, (3, 3), None), pool_strides=((2, 2), None, (2, 2), None),
                  dense_units=(), use_l2_constrained=False, l2_scale=None, noise_stddev=0.02)

    train = np.loadtxt(TRAIN, delimiter=',', skiprows=1)
    train_x = train[:, 1:]
    train_y = train[:, 0]

    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1)) / 255
    train_y = keras.utils.to_categorical(train_y)

    model = create_model(train_x.shape[1:], train_y.shape[-1], param)
    # model.load_weights(model_path)
    print(model.summary())
    model.fit(train_x, train_y, batch_size=1024, epochs=80, validation_split=0.2)
    model.save(os.path.join(RESULTS, 'model.h5'))

    test_x = np.loadtxt(TEST, delimiter=',', skiprows=1)
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1)) / 255
    test_y = model.predict(test_x, batch_size=1024)
    test_y = np.argmax(test_y, axis=1)
    with open(os.path.join(RESULTS, 'submission.csv'), 'w') as f:
        f.write('ImageId,Label\n')
        for i, label in enumerate(test_y, start=1):
            f.write(f'{i},{label}\n')


if __name__ == '__main__':
    main()
