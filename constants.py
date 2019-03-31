import os
import pickle

# CONSTANTS
path_to_data = '/data'
patch_size = 256
max_size = 1024
cropped = True
crop_margin = 100

label_name_to_value = {
    '_background_': 0,
    'door':1,
    'object':2,
    'wall': 3,
    'window': 4,
}

# DERIVATIVES
cropped_str = '_cropped{}'.format(crop_margin) if cropped else ''

def to_name(name):
    return '{name}{cropped_str}_{patch_size}_{max_size}'.format(
        name=name,
        cropped_str=cropped_str,
        patch_size=patch_size,
        max_size=max_size,
    )

def to_file_path(name, ext):
    return "{path_to_data}/{name}.{ext}".format(
        path_to_data=path_to_data,
        name=to_name(name),
        ext=ext,
    )

mean_path = to_file_path('mean', 'p')
mean = pickle.load(open(mean_path, 'rb')) \
        if os.path.exists(mean_path) \
        else [0.49647411704063416, 0.464578241109848, 0.42636236548423767]
std_path = to_file_path('std', 'p')
std = pickle.load(open(std_path, 'rb')) \
        if os.path.exists(std_path) \
        else [0.2456703782081604, 0.2387976497411728, 0.2495083510875702]