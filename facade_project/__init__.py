# Number of images in the dataset
NUM_IMAGES = 418
# Number of rotations generated and saved on disk
NUM_ROTATIONS = 5
# Path to the data where datasets and saved images are stored
PATH_TO_DATA = '/data'
# Max size (either height or width) allowed after resizing the images in order to less computationally expensive
IMG_MAX_SIZE = 1024
# Label names and their respective value in masks
LABEL_NAME_TO_VALUE = {
    'background': 0,
    'wall': 1,
    'window': 2,
    'door': 3,
}
CUT_STEP = 10
CUT_MARGIN = 100

# For the construction of the heatmaps
IS_SIGMA_FIXED = False
SIGMA_FIXED = 20
# When not fixed, widths and heights are used but requires scaling
SIGMA_SCALE = 0.2
HEATMAP_TYPES = ['center']  # , 'width', 'height']
HEATMAP_INCLUDE_MASK = False

DEFAULT_SEED_SPLIT = 238122

# directories for the datasets
FACADE_ROT_DIR = '{}/images/tensor/rotated_rescaled'.format(PATH_TO_DATA)
FACADE_ROT_HEATMAPS_INFOS_PATH = '{}/heatmaps_infos.json'.format(FACADE_ROT_DIR)
FACADE_HEATMAPS_DIR = '{}/heatmaps/tensor/rotated_rescaled'.format(PATH_TO_DATA)
