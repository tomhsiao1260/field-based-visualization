import os

### labels & path
zmin, ymin, xmin, labels = 3513, 1900, 3400, [1, 2]

dirname = f'/Users/yao/Desktop/full-scrolls/community-uploads/yao/scroll1/{zmin:05}_{ymin:05}_{xmin:05}/'

# input directory
volume_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_volume.nrrd')
electrode_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_mask.nrrd')
conductor_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_fiber.nrrd')

# output directory
conductor_x_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_conductor_x.nrrd')
conductor_z_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_conductor_z.nrrd')
potential_init_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_potential_init.nrrd')
potential_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_potential.nrrd')

volume_flatten_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_flatten.nrrd')
electrode_flatten_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_mask_flatten.nrrd')