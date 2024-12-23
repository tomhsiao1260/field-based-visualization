# generate flatten result from potential nrrd data

import os
import sys
import nrrd
import tifffile
import argparse
import numpy as np

# see config_template.py & generate a config.py file
from config import volume_dir, electrode_dir, potential_dir
from config import volume_flatten_dir, electrode_flatten_dir
from app import update_flatten

if __name__ == "__main__":
    ### params
    parser = argparse.ArgumentParser(description='potential transform (for volume & electrode mask)')
    parser.add_argument('--mask', action="store_true", help='False: volume transform, True: electrode transform')
    args = parser.parse_args()

    if not os.path.exists(volume_dir):
        sys.exit(f'volume {os.path.basename(volume_dir)} does not exist')
    if not os.path.exists(electrode_dir):
        sys.exit(f'electrode {os.path.basename(electrode_dir)} does not exist')
    if not os.path.exists(potential_dir):
        sys.exit(f'potential {os.path.basename(potential_dir)} does not exist')

    ### load data
    if (args.mask):
        electrode, header = nrrd.read(electrode_dir)
        potential, header = nrrd.read(potential_dir)

        print('generate flatten electrode ...')
        flatten = update_flatten(electrode, potential)

        print('save flatten electrode ...')
        nrrd.write(electrode_flatten_dir, flatten.astype(np.uint8))
    else:
        volume, header = nrrd.read(volume_dir)
        potential, header = nrrd.read(potential_dir)

        print('generate flatten volume ...')
        flatten = update_flatten(volume, potential)

        print('save flatten volume ...')
        nrrd.write(volume_flatten_dir, flatten.astype(np.uint8))










