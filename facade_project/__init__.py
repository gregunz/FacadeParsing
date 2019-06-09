# Path to the data where datasets and saved images are stored
PATH_TO_DATA = '/data'

# Label names and their respective value in masks
LABEL_NAME_TO_VALUE = {
    'background': 0,
    'wall': 1,
    'window': 2,
    'door': 3,
}

# Number of images in the dataset
NUM_IMAGES = 418
# Number of rotations generated and saved on disk
NUM_ROTATIONS = 5
# Max size (either height or width) allowed after resizing the images in order to less computationally expensive
IMG_MAX_SIZE = 1024
# Default crop size
CROP_SIZE = 768
# Margin to keep when cutting background border of an image
CUT_MARGIN = 50


# For the construction of the heatmaps
IS_SIGMA_FIXED = True
SIGMA_FIXED = 10
# When not fixed, widths and heights are used as sigmas but they require scaling
SIGMA_SCALE = 0.1
# which label to transform into heatmaps
HEATMAP_LABELS = ('window', 'door')
# which types of heatmaps are handled (can be created)
HEATMAP_TYPES_HANDLED = ('center', 'width', 'height')


# Datasets
# default seed for splitting
DEFAULT_SEED_SPLIT = 42

# directories
FACADE_IMAGES_DIR = '{}/images'.format(PATH_TO_DATA)
FACADE_HEATMAPS_DIR = '{}/heatmaps'.format(PATH_TO_DATA)
# coco images
FACADE_COCO_ORIGINAL_DIR = '{}/coco/original'.format(FACADE_IMAGES_DIR)
# labelme images
FACADE_LABELME_ORIGINAL_DIR = '{}/labelme/original'.format(FACADE_IMAGES_DIR)
# random rot images
FACADE_ROT_IMAGES_TENSORS_DIR = '{}/tensor/rotated_rescaled'.format(FACADE_IMAGES_DIR)
# random rot heatmaps
FACADE_ROT_HEATMAPS_TENSORS_DIR = '{}/tensor/rotated_rescaled'.format(FACADE_HEATMAPS_DIR)
FACADE_ROT_HEATMAPS_INFOS_PATH = '{}/json/heatmaps_infos_rotated_rescaled.json'.format(FACADE_HEATMAPS_DIR)

# Facade Random Rotations Dataset statistics
FACADE_ROT_PROPORTIONS = (0.3078, 0.5960, 0.0688, 0.0274)
FACADE_ROT_MEAN = (0.4955, 0.4687, 0.4356)
FACADE_ROT_STD = (0.2254, 0.2239, 0.2373)
