from gen_catalog import stratified_split_unlabeled
from gen_images import sweep_fields, crop_objects_in_rgb
import os

data_dir = os.environ['DATA_PATH']

crop_objects_in_rgb(
    catalog_path='datasets/unlabeled_1.csv',
    input_folder=data_dir + '/dr1/color_images/',
    save_folder=data_dir + '/crops_rgb32/'
)
