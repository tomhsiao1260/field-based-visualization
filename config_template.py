# electrode_label_level_pairs: [(label0, level0), (label1, level1), ...]
# Label: select value in mask.nrrd (that you want it to become electrode)
# Level: electrode horizontal position after flattening (between 0:top ~ 1:bottom)
zmin, ymin, xmin, electrode_label_level_pairs = 3513, 1900, 3400, [(1, 0.15), (2, 0.70)]

dirname = f'/Users/yao/Desktop/full-scrolls/community-uploads/yao/scroll1/{zmin:05}_{ymin:05}_{xmin:05}/'