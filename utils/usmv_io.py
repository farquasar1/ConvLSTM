import fnmatch
import os
import numpy as np
from matplotlib import colors

from PIL import Image


def get_colormap(filename=None):
    """Read a pre-defined colormap from file."""
    if filename is None:
        root = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(root, 'class_rgb.txt')
    colormap = []
    with open(filename, 'r') as f:
        colormap += [list(map(int, l.split())) for l in f.readlines()]
    for i in range(len(colormap), 256):
        colormap += [[i, i, i]]
    # return np.array(colormap, np.float32)
    return np.array(colormap, np.uint8)


def get_plt_colormap(filename=None):
    colormap = get_colormap(filename)
    cmap = np.array(colormap)[:] / 255.0
    colormap = colors.ListedColormap(cmap)
    return colormap


def find_and_read_labels(input_file):
    # Parse input file name
    input_dir, input_name = os.path.split(input_file)
    input_name, ext = os.path.splitext(input_name)
    if ext not in ('.avi', '.mp4'):
        raise IOError('Format %s not supported' % ext)

    # Find matching label file
    label_file = None
    for f in os.listdir(input_dir):
        if fnmatch.fnmatch(f, input_name + '*' + '.labels.tif'):
            label_file = f
            break

    # Read multi-page tif using PIL
    if label_file is None:
        return None, None
        # raise IOError('Label file not found')
    labels = Image.open(os.path.join(input_dir, label_file))
    return label_file, labels


def save_label_image(array, filename):
    # Create and save single frame tiff
    img = Image.fromarray(array)
    img.mode = 'P'
    img.putpalette(get_colormap().ravel())
    img.resize((440, 396)).save(filename)


def read_verified_tag(tags):
    verified = []
    d = tags.as_dict()
    ind = 0
    if 50839 in d and 50838 in d:
        imagej_tags = d[50839]
        tag_sizes = d[50838]
        # skip first two tags
        ind = tag_sizes[0] + tag_sizes[1]
        for j in tag_sizes[2:]:
            if j == 0:
                verified.append(False)
            elif j == 16:
                # TODO: check if string is 'verified'
                # l = imagej_tags[ind:ind+j]
                verified.append(True)
                ind += j
    return verified


def create_final_map(video_file, predictions, output_dir, materials=None):
    import tempfile
    import shutil
    import subprocess
    seq_name = os.path.splitext(os.path.basename(video_file))[0]

    # Save predictions tiff to temp dir
    temp_dir = tempfile.mkdtemp()

    for i, prediction in enumerate(predictions):
        prediction = np.array(prediction, 'uint8')
        filename = os.path.join(temp_dir, '%s_%06d.tif' % (seq_name, i))
        save_label_image(prediction, filename)

    if len(predictions) > 1:
        label_file_copy = os.path.join(temp_dir, seq_name + '.labels.tif')
        # Create tiff image stack and copy final result to output dir
        if materials is not None and os.path.exists(materials):
            materials = os.path.abspath(materials)
        else:
            materials = "UsLiver"
        cmd = 'fiji -cp . Tiff_To_Stack.class -i=%s -o=%s -m=%s' % (temp_dir, label_file_copy, materials)
        local_path = os.path.dirname(os.path.abspath(__file__))
        p = subprocess.Popen(cmd.split(), cwd=os.path.join(local_path, '../bin'))
        ret = p.wait()

        # ret = os.system(cmd)
        if ret != 0:
            print('Could not create stack with command:\n\t\'%s\'' % cmd)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        shutil.copy(label_file_copy, output_dir)
        shutil.rmtree(temp_dir)
