# Number of images in the dataset
NUM_IMAGES = 418
# Number of rotations generated and saved on disk
NUM_ROTATIONS = 5
# Path to the data where datasets and saved images are stored
PATH_TO_DATA = '/data'
# Max size (either height or width) allowed after resizing the images in order to less computationally expensive
IMG_MAX_SIZE = 1024
CROP_SIZE = 768
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
IS_SIGMA_FIXED = True
SIGMA_FIXED = 10
# When not fixed, widths and heights are used but requires scaling
SIGMA_SCALE = 0.2

HEATMAP_LABELS = ['window', 'door']
HEATMAP_TYPES_HANDLED = ['center', 'width', 'height']
HEATMAP_TYPES = ['center']  # , 'width', 'height']
HEATMAP_INCLUDE_MASK = False

DEFAULT_SEED_SPLIT = 238122

# Datasets
FACADE_IMAGES_DIR = '{}/images'.format(PATH_TO_DATA)
FACADE_HEATMAPS_DIR = '{}/heatmaps'.format(PATH_TO_DATA)

FACADE_ROT_IMAGES_TENSORS_DIR = '{}/tensor/rotated_rescaled'.format(FACADE_IMAGES_DIR)
FACADE_ROT_HEATMAPS_TENSORS_DIR = '{}/tensor/rotated_rescaled'.format(FACADE_HEATMAPS_DIR)
FACADE_ROT_HEATMAPS_INFOS_PATH = '{}/json/heatmaps_infos_rotated_rescaled.json'.format(FACADE_HEATMAPS_DIR)

FACADE_ROT_PROPORTIONS = [0.3252, 0.5821, 0.0663, 0.0264]
FACADE_ROT_MEAN = [0.4939, 0.4681, 0.4360]
FACADE_ROT_STD = [0.2271, 0.2261, 0.2410]
