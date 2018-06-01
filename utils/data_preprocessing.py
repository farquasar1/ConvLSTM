import os
import glob
import numpy as np

from utils.sequence_loader import SequenceLoader


def load_splits(splits_file):
    f = open(splits_file, 'r')
    train = {}
    test = {}
    for l in f.read().splitlines():
        l = l.split(',')
        stage, n = l[0].split(':')
        if stage == 'train':
            train[n] = l[1:]
        else:
            test[n] = l[1:]
    return [{'train': train[k], 'test': test[k]} for k in sorted(train.keys())]



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


def list_sequences(seq_dir):
    sequences = []
    for root, dirs, files in os.walk(seq_dir):
        sequences += sorted(filter(lambda v: 'label' not in v, glob.glob(root + '/*tif')))
    print(sequences)
    print(len(sequences))
    return sequences


def load_data(seq_file, with_labels=True, binary=False, start=0, step=2):
    seq = SequenceLoader(seq_file, start=start, step=step, scale=0.25)
    X = []
    y = []
    for i, frame in seq.frames.items():
        # from ugs_utils.usmv_io import get_plt_colormap
        # plt.imshow(frame, cmap='gray')
        # plt.imshow(np.squeeze(seq.labels[i]), cmap=get_plt_colormap(), interpolation='nearest', alpha=0.5, vmin=0, vmax=255)
        # plt.show()
        if with_labels: #  and seq.labels[i].max() > 0:
            X.append(frame)
            y.append(seq.labels[i])
        elif not with_labels:
            X.append(frame)
        else:
            print('unlabelled:', seq_file, i)

    X = np.array(X)
    X = X[:, :96, :108]

    X = np.expand_dims(X, axis=-1)
    if len(y) > 0:
        y = np.array(y)
        y = y[:, :4*96, :4*108]
        if binary:
            y[y>0] = 1
            y = np.expand_dims(y, axis=-1)
        else:
            y = to_categorical(y, 3)
            # y = np.transpose(y, (0, 3, 1, 2))

    X = X.astype('f') / 255
    if not with_labels:
        return X
    return X, y
