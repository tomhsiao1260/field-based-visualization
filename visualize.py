import os
import cv2
import tifffile
import numpy as np

def tif_to_video(video_dir, tif_path, time=15, repeats=1):
    if not os.path.exists(video_dir): os.makedirs(video_dir)

    tif_name = os.path.basename(tif_path)
    video_name = tif_name.replace('.tif', '.mp4')
    video_path = os.path.join(video_dir, video_name)

    # Load tif data
    # data = tifffile.imread(tif_path)
    # data = np.repeat(data, repeats=repeats, axis=0)
    # data = np.repeat(data, repeats=repeats, axis=1)
    # data = np.repeat(data, repeats=repeats, axis=2)
    # d, h, w = data.shape[:3]

    data_1 = tifffile.imread('extracted_data_.tif')
    pad = np.zeros((data_1.shape[0], data_1.shape[1], 134))
    data_1 = np.concatenate((pad, data_1, pad), axis=2)

    data_2 = tifffile.imread('03513_01900_03400_volume.tif')[:, 170:520, :]

    empty = np.zeros((data_1.shape[0], 5, data_1.shape[2]))
    data = np.concatenate((empty, data_1, empty, data_2), axis=1)
    d, h, w = data.shape[:3]
    data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

    # Create video writer
    fps = d / time
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    # Save as video
    for layer in range(d):
        image = data[layer, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow(tif_name, image)
        cv2.waitKey(int(1000 / fps))
        cv2.destroyAllWindows()
        writer.write(image)
    writer.release()

if __name__ == '__main__':
    tif_to_video('./', 'extracted_data_.tif')