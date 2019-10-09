from typing import Tuple, Optional
import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
    def __init__(
            self,
            conv_filters: Tuple[int, ...],
            kernel_sizes: Tuple[Tuple[int, int], ...],
            strides: Tuple[Tuple[int, int], ...],
            pool_sizes: Tuple[Optional[Tuple[int, int]], ...],
            pool_strides: Tuple[Optional[Tuple[int, int]], ...],
            conv_dropout_rates: Tuple[Optional[float], ...],
            dense_units: Tuple[int, ...],
            dense_dropout_rates: Tuple[Optional[float], ...],
            l2_constrained_scale: Optional[float],
            center_loss_margin: Optional[float],
            noise_stddev: Optional[float]):
        assert len(conv_filters) == len(kernel_sizes) == len(strides) \
               == len(pool_sizes) == len(pool_strides) == len(conv_dropout_rates)
        assert len(dense_units) == len(dense_dropout_rates)
        for pool_size, pool_stride in zip(pool_sizes, pool_strides):
            is_size_none, is_stride_none = pool_size is None, pool_stride is None
            assert not (is_size_none ^ is_stride_none)
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.conv_dropout_rates = conv_dropout_rates
        self.dense_units = dense_units
        self.dense_dropout_rates = dense_dropout_rates
        self.l2_constrained_scale = l2_constrained_scale
        self.center_loss_margin = center_loss_margin
        self.noise_stddev = noise_stddev


def create_model(input_shape: Tuple[int, ...], output_shape: int, param: Param) -> keras.Model:
    inputs = keras.Input(input_shape)
    x = inputs
    if param.noise_stddev is not None:
        x = layers.GaussianNoise(param.noise_stddev)(x)
    x = layers.Lambda(lambda z: z - K.mean(z, axis=1, keepdims=True))(x)
    # x = layers.Lambda(lambda z: z / K.sqrt(K.var(z, axis=1, keepdims=True)))(x)
    for i in range(len(param.conv_filters)):
        x = layers.Conv2D(param.conv_filters[i], kernel_size=param.kernel_sizes[i], strides=param.strides[i],
                          padding='same')(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.ELU()(x)
        if param.pool_sizes[i] is not None:
            x = layers.MaxPooling2D(pool_size=param.pool_sizes[i], strides=param.pool_strides[i])(x)
        if param.conv_dropout_rates[i] is not None:
            x = layers.Dropout(param.conv_dropout_rates[i])(x)
    x = layers.Flatten()(x)

    for units, dropout_rate in zip(param.dense_units, param.dense_dropout_rates):
        x = layers.Dense(units, activation='elu')(x)
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate)(x)

    if param.l2_constrained_scale:
        scale = param.l2_constrained_scale
        x = layers.Lambda(lambda z: K.l2_normalize(z, axis=1) * scale)(x)
        outputs = layers.Dense(output_shape, kernel_constraint=keras.constraints.UnitNorm(),
                               use_bias=False)(x)
    else:
        outputs = layers.Dense(output_shape)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    if param.center_loss_margin:
        loss = CenterLoss(param.center_loss_margin)
    else:
        loss = tf.losses.softmax_cross_entropy
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
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

    for i in range(len(param.conv_filters)):
        x_re, x_im = complexize_kernel(layers.Conv2D, param.conv_filters[i], kernel_size=param.kernel_sizes[i],
                                       strides=param.strides[i],
                                       padding='same', activation=layers.Activation("tanh"))(x_re, x_im)
        x_re = layers.BatchNormalization(axis=-1)(x_re)
        x_im = layers.BatchNormalization(axis=-1)(x_im)
        if param.pool_sizes[i] is not None:
            pool_size = param.pool_sizes[i]
            pool_strides = param.pool_strides[i]
            x_re = layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides)(x_re)
            x_im = layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides)(x_im)
        if param.conv_dropout_rates[i] is not None:
            dropout_rate = param.conv_dropout_rates[i]
            x_re = layers.Dropout(dropout_rate)(x_re)
            x_im = layers.Dropout(dropout_rate)(x_im)
    x_re = layers.Flatten()(x_re)
    x_im = layers.Flatten()(x_im)
    for units, dropout_rate in zip(param.dense_units, param.dense_dropout_rates):
        x_re, x_im = complexize_kernel(layers.Dense, units, activation=layers.Activation("tanh"))(x_re, x_im)
        if dropout_rate is not None:
            x_re = layers.Dropout(dropout_rate)(x_re)
            x_im = layers.Dropout(dropout_rate)(x_im)
    # x = layers.Lambda(lambda d: K.sqrt(K.square(d[0]) + K.square(d[1])))([x_re, x_im])
    if param.l2_constrained_scale:
        x = layers.Lambda(lambda z: K.l2_normalize(z, axis=1) * param.l2_constrained_scale)(x_re)
        outputs = layers.Dense(output_shape, kernel_constraint=keras.constraints.UnitNorm(),
                               use_bias=False)(x)
    else:
        outputs = layers.Dense(output_shape)(x_re)
    model = keras.Model(inputs=inputs, outputs=outputs)
    if param.center_loss_margin:
        loss = CenterLoss(param.center_loss_margin)
    else:
        loss = tf.losses.softmax_cross_entropy
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model


def main():
    param = Param(conv_filters=(16, 16, 32, 32, 64, 64), kernel_sizes=((5, 5),) * 6, strides=((1, 1),) * 6,
                  conv_dropout_rates=(0.25,) * 6,
                  pool_sizes=(None, (3, 3), None, (3, 3), None, (3, 3)),
                  pool_strides=(None, (2, 2), None, (2, 2), None, (3, 3)),
                  dense_units=(128,), dense_dropout_rates=(0.5,),
                  l2_constrained_scale=0.5, center_loss_margin=0.1, noise_stddev=0.02)

    train = np.loadtxt(TRAIN, delimiter=',', skiprows=1)
    train_x = train[:, 1:]
    train_y = train[:, 0]

    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1)) / 255
    train_y = keras.utils.to_categorical(train_y)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='constant')
    datagen.fit(train_x)

    model = create_model(train_x.shape[1:], train_y.shape[-1], param)
    # model.load_weights(model_path)
    print(model.summary())
    model.fit_generator(
        datagen.flow(train_x, train_y, batch_size=1024),
        validation_data=[val_x, val_y],
        epochs=80, steps_per_epoch=train_x.shape[0] // 1024)
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
