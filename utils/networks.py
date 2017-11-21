from keras.models import Sequential, Model
from keras.layers import (ConvLSTM2D, BatchNormalization, Convolution3D, Convolution2D,
                          TimeDistributed, MaxPooling2D, UpSampling2D, Input, merge)

from ugs_utils.keras_extensions import (categorical_crossentropy_3d_w, softmax_3d, softmax_2d)


def binary_net(input_shape):
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation='sigmoid',
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss='binary_crossentropy', optimizer='adadelta')
    return net


def class_net(input_shape):
    c = 3
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=4 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=8 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')
    return net


def class_net_ms(input_shape):
    c = 12
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, input_shape=input_shape,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(TimeDistributed(MaxPooling2D((2, 2), (2, 2))))

    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    # net.add(TimeDistributed(MaxPooling2D((2, 2), (2, 2))))

    net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    # net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
    #                    border_mode='same', return_sequences=True))
    # net.add(TimeDistributed(UpSampling2D((2, 2))))
    # net.add(ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3,
    #                    border_mode='same', return_sequences=True))

    net.add(TimeDistributed(UpSampling2D((2, 2))))
    net.add(Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                          border_mode='same', dim_ordering='tf'))
    net.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')
    return net


def class_net_fcn_1p(input_shape):
    c = 12
    input_img = Input(input_shape, name='input')

    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(input_img)
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
    c1 = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    c2 = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(c2)
    x = merge([c1, x], mode='concat')
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
    output = Convolution3D(nb_filter=3, kernel_dim1=1, kernel_dim2=3,
                           kernel_dim3=3, activation=softmax_3d(class_dim=-1),
                           border_mode='same', dim_ordering='tf', name='output')(x)
    model = Model(input_img, output=[output])
    model.compile(loss=categorical_crossentropy_3d_w(4, class_dim=-1), optimizer='adadelta')

    return model


def class_net_fcn_1p_lstm(input_shape):
    c = 12
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c2)
    x = merge([c1, x], mode='concat')

    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    # x = TimeDistributed(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 396, 440), border_mode='valid'))(x)

    output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)

    model = Model(input_img, output=[output])
    model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adadelta')
    return model


def class_net_fcn_2p_lstm(input_shape):
    c = 12
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c3 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = merge([c2, x], mode='concat')
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = merge([c1, x], mode='concat')
    # x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)

    model = Model(input_img, output=[output])
    model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adadelta')
    return model
