from pathlib import Path
import numpy as np
import cv2
import bz2
import pickle
import _pickle as cPickle
import json


def load_cam_json(file):
    with open(file, "r") as f:
        data = json.load(f)

    K = np.array(data["K"]).reshape(3, 3)
    RT = np.array(data["RT"]).reshape(3, 4)

    return K, RT, data


# Saves the "data" with the "title" and adds the .pickle
def dump_pickle(title, data):
    # print(f'Write to {title}.pickle')
    pikd = open(title + ".pickle", "wb")
    pickle.dump(data, pikd)
    pikd.close()


# loads and returns a pickled objects
def load_pickle(file):
    pikd = open(file, "rb")
    data = pickle.load(pikd)
    pikd.close()
    return data


# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', "w") as f:
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data


def read_img(file, flags=cv2.IMREAD_COLOR):
    """Return image with range [0.0, 1.0] and RGB mode"""
    assert Path(file).is_file()
    img = cv2.imread(file, flags=flags)
    if img is None:
        return None
    else:
        img = img.astype(np.float) / 255.0

    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        return img
    else:
        raise ValueError(f'Wrong image shape')


def write_img(file, img):
    """Write image to file
    Notes: image with range [0.0, 1.0] or [0, 255] and RGB mode
    """
    if np.max(img) < 2.01:
        img = img * 255.0

    cv2.imwrite(file, img[:, :, ::-1])


def resize_img(img, img_height, img_width):
    return cv2.resize(img, (img_height, img_width))


def export_ply_v_vc_f(out, v, vc, f):
    """Export vertices, vertices colors and faces to ply file
    Notes:
        - Input faces start from index 0
    """
    with open(out, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('element vertex ' + str(v.shape[0]) + '\n')
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write(f'element face {f.shape[0]}\n')
        fout.write(f'property list uchar int vertex_indices\n')
        fout.write('end_header\n')

        # write vertex
        for i in range(v.shape[0]):
            cur_color = vc[i]
            fout.write(f'{v[i, 0]} '
                       f'{v[i, 1]} '
                       f'{v[i, 2]} '
                       f'{int(cur_color[0] * 255)} '
                       f'{int(cur_color[1] * 255)} '
                       f'{int(cur_color[2] * 255)}'
                       f'\n')

        # write face
        for i in range(f.shape[0]):
            fout.write(f'3 '
                       f'{f[i, 0]} '
                       f'{f[i, 1]} '
                       f'{f[i, 2]}'
                       f'\n')


def export_ply(out, v):
    with open(out, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('element vertex ' + str(v.shape[0]) + '\n')
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('end_header\n')

        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))


def get_list_from_file(in_file):
    with open(in_file, 'r') as fin:
        lines = fin.readlines()
    return [item.strip() for item in lines]


def save_list_to_file(data, out_file):
    with open(out_file, 'w') as fout:
        for item in data:
            fout.write(f'{item}\n')



