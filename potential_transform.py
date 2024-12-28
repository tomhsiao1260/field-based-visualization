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
from potential_generate import update_flatten

if __name__ == "__main__":
    ### params
    parser = argparse.ArgumentParser(description='potential transform (for volume & electrode mask)')
    parser.add_argument('--z', type=int, help='z index')
    parser.add_argument('--y', type=int, help='y index')
    parser.add_argument('--x', type=int, help='x index')
    parser.add_argument('--mask', action="store_true", help='False: volume transform, True: electrode transform')
    args = parser.parse_args()

    zmin, ymin, xmin = args.z, args.y, args.x

    volume_path = volume_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
    electrode_path = electrode_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
    potential_path = potential_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
    volume_flatten_path = volume_flatten_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
    electrode_flatten_path = electrode_flatten_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)

    if not os.path.exists(volume_path):
        sys.exit(f'volume {os.path.basename(volume_path)} does not exist')
    if not os.path.exists(electrode_path):
        sys.exit(f'electrode {os.path.basename(electrode_path)} does not exist')
    if not os.path.exists(potential_path):
        sys.exit(f'potential {os.path.basename(potential_path)} does not exist')

    ### load data
    if (args.mask):
        electrode, header = nrrd.read(electrode_path)
        potential, header = nrrd.read(potential_path)

        print('generate flatten electrode ...')
        flatten = update_flatten(electrode, potential)

        print('save flatten electrode ...')
        nrrd.write(electrode_flatten_path, flatten.astype(np.uint8))
    else:
        volume, header = nrrd.read(volume_path)
        potential, header = nrrd.read(potential_path)

        print('generate flatten volume ...')
        flatten = update_flatten(volume, potential)

        print('save flatten volume ...')
        nrrd.write(volume_flatten_path, flatten.astype(np.uint8))










