# Number of images in the dataset
NUM_IMAGES = 418
# Number of rotations generated and saved on disk
NUM_ROTATIONS = 5
# Path to the data where datasets and saved images are stored
PATH_TO_DATA = '/data'
# Size of the patches used for albunet
PATCH_SIZE = 256
# Max size (either height or width) allowed after resizing the images in order to less computationally expensive
IMG_MAX_SIZE = 1024
# Label names and their respective value in masks
LABEL_NAME_TO_VALUE = {
    'background': 0,
    'wall': 1,
    'window': 2,
    'door': 3,
}
# Variables used for naming when generation data or to store model names depending on the configuration
IS_USING_CROP = True
CUT_STEP = 10
CUT_MARGIN = 100

# For the construction of the heatmaps
SIGMA_FIXED = True
SIGMA_SCALE = 4
SIGMA = 50



def to_name(name):
    cropped_str = '_cropped{}'.format(CUT_MARGIN) if IS_USING_CROP else ''
    return '{name}{cropped_str}_{patch_size}_{max_size}'.format(
        name=name,
        cropped_str=cropped_str,
        patch_size=PATCH_SIZE,
        max_size=IMG_MAX_SIZE,
    )
