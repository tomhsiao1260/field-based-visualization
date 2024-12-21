# generate flatten result from potential nrrd data

import os
import sys
import nrrd
import tifffile
import numpy as np

# see config_template.py & generate a config.py file
from config import zmin, ymin, xmin, dirname
from app import update_flatten

if __name__ == "__main__":
    # input
    volume_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_volume.nrrd')
    potential_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_potential.nrrd')

    # output
    flatten_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_flatten.nrrd')

    if not os.path.exists(volume_dir):
        sys.exit(f'volume {os.path.basename(volume_dir)} does not exist')
    if not os.path.exists(potential_dir):
        sys.exit(f'potential {os.path.basename(potential_dir)} does not exist')

    ### load data
    volume, header = nrrd.read(volume_dir)
    potential, header = nrrd.read(potential_dir)
    d, h, w = volume.shape

    print('generate flatten result ...')
    flatten = update_flatten(volume, potential)

    print('save flatten result ...')
    nrrd.write(os.path.join(dirname, f"{zmin:05}_{ymin:05}_{xmin:05}_flatten.nrrd"), flatten.astype(np.uint8))
    tifffile.imwrite(os.path.join(dirname, f"{zmin:05}_{ymin:05}_{xmin:05}_flatten.tif"), flatten.astype(np.uint8))










