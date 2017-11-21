import numpy as np
from keras import backend as K


def to_categorical(y, nb_classes=None):
    """ Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    :param y: label image with values between 0 and 255 (ex. shape: (h, w))
    :param nb_classes: number of classes
    :return: "binary" label image with as many channels as number of labels (ex. shape: (h, w, #lab))
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y) + 1

    yy = y.flatten()
    Y = np.zeros((len(yy), nb_classes))
    for i in range(len(yy)):
        Y[i, yy[i]] = 1.
    Y = np.reshape(Y, list(y.shape) + [nb_classes])
    return Y


def softmax_3d(class_dim=-1):
    """ 3D extension of softmax, class is last dim"""
    def activation(x):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=class_dim, keepdims=True))
            s = K.sum(e, axis=class_dim, keepdims=True)
            return e / s
        else:
            raise Exception('Cannot apply softmax to a tensor that is not 2D or '
                            '3D. Here, ndim=' + str(ndim))
    return activation


def categorical_crossentropy_3d_w(alpha, class_dim=-1):
    """ Weighted 3D extension CCE, class is last dim"""
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = y_true * K.log(y_pred)
        cce = K.sum(cce, axis=class_dim)
        cce *= 1 + alpha * K.clip(K.cast(K.argmax(y_true, axis=class_dim), K.floatx()), 0, 1)
        cce = -K.sum(K.sum(cce, axis=-1), axis=-1)
        return cce
    return loss


def categorical_crossentropy_3d(class_dim=-1):
    """2D categorical crossentropy loss
    """
    def loss(y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        # avoid numerical instability with _EPSILON clipping
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = -K.sum(K.sum(K.sum(y_true * K.log(y_pred), axis=class_dim), axis=-1), axis=-1)
        return cce
    return loss


def softmax_2d(class_dim=-1):
    """2D softmax activation
    """
    def activation(x):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=class_dim, keepdims=True))
            s = K.sum(e, axis=class_dim, keepdims=True)
            return e / s
        else:
            raise Exception('Cannot apply softmax to a tensor that is not 2D or '
                            '3D. Here, ndim=' + str(ndim))
    return activation


def categorical_crossentropy_2d(class_dim=-1):
    """2D categorical crossentropy loss
    """
    def loss(y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        # avoid numerical instability with _EPSILON clipping
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = -K.sum(K.sum(y_true * K.log(y_pred), axis=class_dim), axis=-1)
        return cce
    return loss


def categorical_crossentropy_2d_w(alpha, class_dim=-1):
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=class_dim, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cce = y_true * K.log(y_pred)
        cce = K.sum(cce, axis=class_dim)
        cce *= 1 + alpha * K.clip(K.argmax(y_true, axis=class_dim), 0, 1)
        cce = -K.sum(cce, axis=-1)
        return cce
    return loss
