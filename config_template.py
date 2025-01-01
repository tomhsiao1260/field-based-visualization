import os

### example

# python potential_init.py --z 1200 --y 2305 --x 3490 --labels 2 --plot
# python potential_generate.py --z 1200 --y 2305 --x 3490 --labels 2 --plot

# python potential_transform.py --z 1200 --y 2305 --x 3490
# python potential_transform.py --z 1200 --y 2305 --x 3490 --mask

# python ppm.py --z 1200 --y 2305 --x 3490 --zo 560 --yo 200 --xo 300

# python ppm.py --z 6039 --y 1783 --x 4008 --zf 322 --yf 422 --xf 750

# data directory
dirname = (
    "/Users/yao/Desktop/full-scrolls/community-uploads/yao/scroll1/{:05}_{:05}_{:05}/"
)

# input directory
volume_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_volume.nrrd")
electrode_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_mask.nrrd")
conductor_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_fiber.nrrd")

# output directory
conductor_x_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_conductor_x.nrrd")
conductor_z_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_conductor_z.nrrd")
potential_init_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_potential_init.nrrd")
potential_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_potential_v1.nrrd")

volume_flatten_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_flatten.nrrd")
electrode_flatten_dir = os.path.join(dirname, "{:05}_{:05}_{:05}_mask_flatten.nrrd")
