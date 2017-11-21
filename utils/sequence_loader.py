import fnmatch
import os
from copy import deepcopy

from PIL import Image
import subprocess
import tempfile
import shutil
from .usmv_io import save_label_image
import numpy as np
from scipy.misc import imresize

from .usmv_io import get_colormap
from .usmv_io import read_verified_tag
from distutils.version import LooseVersion
from collections import OrderedDict


class SequenceLoader:
    def __init__(self, input_file, default_height=-1, crop_width=-1, roi=None, scale=-1, mask=None, start=0, step=1):
        self.width = 0
        self.height = default_height
        self.frames = OrderedDict()
        self.labels = OrderedDict()
        self.crop_width = crop_width
        self.label_set = []
        self.labels_tag = None
        self.roi = roi
        self.scale = scale
        self.mask = mask

        # Parse input file name
        input_dir, input_name = os.path.split(input_file)
        self.input_name, ext = os.path.splitext(input_name)
        if ext not in ('.avi', '.mp4', '.tif'):
            raise IOError('Format %s not supported' % (ext))

        # Find matching label file
        label_file = None
        for f in os.listdir(input_dir):
            if fnmatch.fnmatch(f, self.input_name + '' + '.labels.tif'):
                label_file = f
                break

        # Read multi-page tif using PIL
        if label_file is None:
            raise IOError('Label file not found')

        # Read video
        if ext in ('.avi', '.mp4'):
            import cv2
            video = cv2.VideoCapture(input_file)
            if LooseVersion(cv2.__version__) < LooseVersion('3'):
                num_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            else:
                num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(num_frames):
                is_valid, img = video.read()
                if not is_valid:
                    print('Cannot read frame: %d of %s' % (i, input_file))
                    num_frames = i + 1
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self._add_frame(i, img)
        else:  # TIFF
            seq = Image.open(input_file)
            num_frames = seq.n_frames
            for i in range(start, seq.n_frames, step):
                seq.seek(i)
                a = np.array(seq.convert('L'))
                self._add_frame(i, a)

        # Read labels
        labels = Image.open(os.path.join(input_dir, label_file))
        self.label_file_fullpath = os.path.join(input_dir, label_file)
        self.label_file = label_file
        self.labels_tag = deepcopy(labels.tag)
        self.nb_classes = 0
        for i in range(start, num_frames, step):
            labels.seek(i)
            cls = np.array(labels)
            self.nb_classes = max(self.nb_classes, np.max(cls) + 1)
            # Read 'verified' tag in first frame
            if i == 0:
                try:
                    self.verified = read_verified_tag(labels.tag)
                except:
                    pass
                # if len(self.verified) == 0:
                #     raise IOError('Could not read verified tag.')
            self._add_labels(i, cls)

    def _process_roi(self, data, crop=True, mask=True, label=False):
        if self.scale > 0:
            import scipy.ndimage
            if label:
                pass
                # data = scipy.ndimage.zoom(data, self.scale, order=0)
            else:
                data = scipy.ndimage.zoom(data, self.scale, order=3)

        if self.roi is None:
            return data
        else:
            _data = data

        x0 = self.roi['x0']
        y0 = self.roi['y0']
        width = self.roi['width']
        height = self.roi['height']

        if mask and self.mask is not None:
            _data = _data * self.mask

        if x0 == 0 and y0 == 0 and width == data.shape[1] and height == data.shape[0]:
            return _data

        # Pad with zeros if the roi is out of bounds
        if x0 < 0 or x0 + width > data.shape[1] or y0 < 0 or y0 + height > data.shape[0]:
            pad_x = (max(-x0, 0), max(x0 + width - data.shape[1], 0))
            pad_y = (max(-y0, 0), max(y0 + height - data.shape[0], 0))
            _data = np.pad(_data, (pad_y, pad_x), 'constant', constant_values=(0, 0))
            x0 += pad_x[0]
            y0 += pad_y[0]

        # Crop or center the data
        if crop:
            _data = _data[y0:y0 + height, x0:x0 + width]
        else:
            dx = int(_data.shape[1] / 2.0 - (x0 + width / 2.0))
            dy = int(_data.shape[0] / 2.0 - (y0 + height / 2.0))
            from scipy.ndimage.interpolation import shift
            _data = shift(_data, (dy, dx), cval=0.0)
        return _data

    def _resize(self, data, interpolation='lanczos'):
        if self.height <= 0:
            width = data.shape[1]
            height = data.shape[0]
        else:
            height = self.height
            width = data.shape[1] * height / data.shape[0]
        if width != data.shape[1] or height != data.shape[0]:
            _data = imresize(data, (width, height), interpolation)
            if 0 < self.crop_width < width:
                dx = int(width - self.crop_width)
                _data = _data[:, dx / 2:dx / 2 + self.crop_width]
            elif self.crop_width > 0 and width < self.crop_width:
                dx = int(self.crop_width - width)
                pad_x = (dx / 2, (dx + 1) / 2)
                pad_y = (0, 0)
                _data = np.pad(_data, (pad_y, pad_x), 'constant', constant_values=(0, 0))
            return _data
        else:
            return data

    def _add_frame(self, i, frame):
        frame = self._process_roi(frame)
        self.frames[i] = self._resize(frame)

    def _add_labels(self, i, cls):
        _cls = self._process_roi(cls, crop=True, mask=False, label=True)
        _cls = self._resize(_cls, interpolation='nearest')
        self.labels_tag[256] = _cls.shape[1]
        self.labels_tag[257] = _cls.shape[0]
        self.labels[i] = _cls

    def merge_labels(self, merged_labels):
        label_set = np.unique(merged_labels).tolist()
        self.nb_classes = 0
        for k, label in self.labels.iteritems():
            for old_label, new_label in enumerate(merged_labels):
                label[label == old_label] = label_set.index(new_label)
            self.labels[k] = label
            self.nb_classes = max(self.nb_classes, np.max(label) + 1)
        self.label_set = label_set


    @property
    def nb_elements(self):
        return len(self.frames)

    @property
    def colormap(self):
        colormap = get_colormap()
        # if len(self.label_set) > 0:
        #     colormap = colormap[self.label_set]
        return colormap

    def get_all_tags(self):
        return self.nb_elements * [self.labels_tag]

    def save(self, output_dir):
        import cv2
        # save video
        height, width = self.frames[0].shape[:2]
        # fourcc = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')
        writer = cv2.VideoWriter(os.path.join(output_dir, self.input_name + '.avi'), 0, 24.0, (width, height))
        for frame in self.frames.values():
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        writer.release()

        from tifffile import imsave
        imsave(os.path.join(output_dir, self.input_name + '.tif'),
               np.array(list(self.frames.values())))

        # labels
        temp_dir = tempfile.mkdtemp()
        label_file_copy = os.path.join(temp_dir, 'tmp.labels.tif')

        tags = self.get_all_tags()
        for i, l in self.labels.items():
            filename = os.path.join(temp_dir, '%s_%06d.tif' % ('tmp', i))
            save_label_image(l, filename)

        cmd = 'fiji -cp . Tiff_To_Stack.class -i=%s -o=%s -m=%s' % (temp_dir, label_file_copy, 'label_set.dat')
        local_path = os.path.dirname(os.path.abspath(__file__))
        p = subprocess.Popen(cmd.split(), cwd=os.path.join(local_path, '../bin'))
        ret = p.wait()

        # ret = os.system(cmd)
        if ret != 0:
            print('Could not create stack with command:\n\t\'%s\'' % cmd)

        shutil.copy(label_file_copy, os.path.join(output_dir, self.label_file))
        shutil.rmtree(temp_dir)
