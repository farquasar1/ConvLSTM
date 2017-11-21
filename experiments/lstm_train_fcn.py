import matplotlib
matplotlib.use('agg')

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
K.set_image_dim_ordering('tf')

import matplotlib.pyplot as plt
import numpy as np
from utils.networks import class_net_fcn_2p_lstm
from utils.data_preprocessing import list_sequences, load_data, load_splits


def net_summary(net):
    import sys
    from io import StringIO
    # Temporarily redirect stdout, print net summary and then restore stdout
    msg = StringIO()
    out = sys.stdout
    sys.stdout = msg
    net.summary()
    sys.stdout = out
    return msg.getvalue()


def seq_augment(seq_X, seq_y):
    # TODO: ugly hack, consider to rewrite the sequence generator so that it fits with hal
    from hal.augmentation.geometric_transforms import random_rigid_transform, apply_geometric_transform
    height, width = seq_y.shape[1:3]
    _y, _x = np.mgrid[:height, :width].astype('f')
    _x, _y = random_rigid_transform(_x, _y, scale=0.1, shift=(5.0, 5.0))
    for k in range(seq_y.shape[0]):
        frame = seq_y[k, ...].transpose([2, 0, 1])
        frame = apply_geometric_transform(frame, _x, _y, interp='nearest')
        seq_y[k, ...] = frame.transpose([1, 2, 0])
    from scipy import ndimage
    _x = .25 * ndimage.zoom(_x, 0.25, order=3)
    _y = .25 * ndimage.zoom(_y, 0.25, order=3)
    for k in range(seq_X.shape[0]):
        frame = seq_X[k, ...].transpose([2, 0, 1])
        frame = apply_geometric_transform(frame, _x, _y, interp='cubic')
        seq_X[k, ...] = frame.transpose([1, 2, 0])
    return seq_X, seq_y


class SequenceGenerator:

    def __init__(self, sequences, seq_length, seq_per_seq=50, shuffle=True, step=5, jitter=None, augment=None):
        self.seq_length = seq_length
        self.step = step
        self.Xs = []
        self.ys = []
        self.ys1 = []
        for s in sequences:
            print(s)
            X, y = load_data(s, start=0, step=1)
            self.Xs.append(X)
            self.ys.append(y)
            self.ys1.append(y)
        self.nb_elements = seq_per_seq * len(sequences)
        self.shuffle = shuffle
        self.jitter = jitter
        self.augment = augment
        print(self.nb_elements)

    def generate_batch(self, batch_size):
        for ib in range(self.nb_elements // batch_size):
            X = []
            y = []
            y1 = []
            for j in range(batch_size):
                if self.shuffle:
                    i = np.random.randint(0, len(self.Xs))
                else:
                    i = 0
                seq_X = self.Xs[i]
                seq_y = self.ys[i]
                seq_y1 = self.ys1[i]
                if self.shuffle:
                    s = np.random.randint(0, len(seq_X) - self.seq_length * self.step)
                else:
                    s = ib
                indices = np.arange(s, s + self.seq_length * self.step, self.step)
                if self.jitter is not None:
                    # Perturb indices by introducing random time jitter
                    indices += np.random.randint(-self.jitter, self.jitter + 1, indices.shape)
                    # Clip to avoid out of bound indices
                    indices = np.clip(indices, 0, len(seq_X) - self.seq_length * self.step)
                seq_X = seq_X[indices]
                seq_y = seq_y[indices]
                seq_y1 = seq_y1[indices]
                if self.augment:
                    seq_X, seq_X = seq_augment(seq_X, seq_X)
                X.append(seq_X)
                y.append(seq_y)
                y1.append(seq_y1)
            X = np.array(X).astype('f') / 255
            y = np.array(y)
            y1 = np.array(y1)
#            print("\n",X.shape,y.shape,y1.shape)
            yield Batch({'input': X}, {'ys1': y1})

    def nb_batches(self, batch_size):
        return self.nb_elements // batch_size


class ModelCheckpointsCallback(Callback):
    def __init__(self, net, output_dir):
        super().__init__()
        self.net = net
        self.output_dir = output_dir
        self.training_loss_history = []
        self.validation_loss_history = []

    def on_end_epoch(self):
        if len(self.validation_loss_history) > 0:
            if self.state.validation_loss < np.min(self.validation_loss_history):
                self.net.save_weights('%s/weights_%06d_%d.h5' % (self.output_dir, self.state.epoch, int(self.state.validation_loss)))

        self.training_loss_history.append(self.state.training_loss)
        self.validation_loss_history.append(self.state.validation_loss)


def generate_results(net, generator, output_dir, out_prefix):
    for i, b in enumerate(generator.generate_batch(1)):
        img = [x for x in np.squeeze(b.input)]
        t = [x for x in np.squeeze(b.output.astype('f'))]
        r = [x for x in np.squeeze(net.predict(b.input))]
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(np.hstack(img), cmap='gray')
        plt.subplot(3, 1, 2)
        plt.imshow(np.hstack(t), cmap='gray')
        plt.subplot(3, 1, 3)
        plt.imshow(np.hstack(r), cmap='gray')
        plt.savefig('%s/%s_%03d.png' % (output_dir, out_prefix, i))
        plt.close()

        
def train_model(network, sequences, sequences_test, nb_epochs=100, seq_length=10, seq_step=5, seq_per_seq=10,
                output_dir=None, file_suffix='', jitter=None, augment=None):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Setup data generators
    train = SequenceGenerator(sequences, seq_length, seq_per_seq=seq_per_seq, step=seq_step, jitter=jitter, augment=augment)
    val = SequenceGenerator(sequences_test, seq_length, seq_per_seq=seq_per_seq, step=seq_step)
    input_shape = [None, ] + list(train.Xs[0].shape[1:])
    a=train.generate_batch(10)    
    for i in a:
        print (i)
    #next(a)
    
    # Setup model and train
    model = network(input_shape)
    print(net_summary(net))
    model.fit(train, nb_epochs=nb_epochs, batch_size=1,val_data_generator=val)
    loss = model._state.validation_loss

    # Save output
    if output_dir is not None:
        net.save_weights(os.path.join(output_dir, 'weights' + file_suffix + '.h5'), overwrite=True)
        plt.ylim([0, 100000])
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss' + file_suffix + '.png'))
        plt.close()
        generate_results(net, train, output_dir, 'train' + file_suffix)
        generate_results(net, val, output_dir, 'val' + file_suffix)

    return net, loss

if __name__ == '__main__':
    # Load data split
    input_dir = os.path.expanduser('~/Data/')
    split = load_splits('splits.txt')[0]
    sequences = [os.path.join(input_dir, f) for f in split['train']]
    sequences_test = [os.path.join(input_dir, f) for f in split['test']]

    network = class_net_fcn_2p_lstm  # input_shape = (None, 96, 108, 1)

    train_model(network, sequences, sequences_test, seq_length=10, seq_step=5, output_dir='tmp')
