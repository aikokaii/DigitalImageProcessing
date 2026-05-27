import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import random

base_dir = os.path.dirname(__file__)
folder_name = 'dataset'
path_to_dataset = os.path.join(base_dir, folder_name)

def read_pgm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().strip()
        if header != b'P5':
            raise ValueError(f"File {file_path} bukan format P5 PGM.")
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(pos)
                break
        dim_line = f.readline().strip().split()
        width, height = int(dim_line[0]), int(dim_line[1])
        maxval = int(f.readline().strip())
        return np.fromfile(f, dtype=np.uint8).reshape((height, width))

def equalize_hist(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf = np.ma.filled(cdf_masked, 0).astype(np.uint8)
    return cdf[img]

def convolve2d(image, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(k_h):
        for j in range(k_w):
            output += padded[i : i + image.shape[0], j : j + image.shape[1]] * kernel[i, j]
    return output

def gaussian_blur(img):
    kernel = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ], dtype=np.float64) / 273.0
    blurred = convolve2d(img, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)

def laplacian(img):
    kernel = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float64)
    laplacian_img = convolve2d(img, kernel)
    return np.clip(np.absolute(laplacian_img), 0, 255).astype(np.uint8)

def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return table[img]

def load_and_process_dataset(path_folder):
    file_pattern = os.path.join(path_folder, "*.pgm")
    file_list = glob.glob(file_pattern)
    file_list.sort()

    if not file_list:
        print(f"File not found in: {path_folder}")
        return

    print(f"Succesfull find {len(file_list)} file .pgm!")

    num_samples = min(5, len(file_list))
    samples = random.sample(file_list, num_samples)

    num_cols = 4
    plt.figure(figsize=(18, 10))

    for i, file_path in enumerate(samples):
        try:

            img = read_pgm(file_path)

            img_equ = equalize_hist(img)
            img_blur = gaussian_blur(img_equ)
            img_laplacian = laplacian(img_blur)
            img_gamma_bright = gamma_correction(img, gamma=0.45)

            display_list = [img, img_equ, img_blur, img_laplacian, img_gamma_bright]
            titles = ['original', 'equalized', 'gaussian blur', 'laplacian', 'gamma gelap']

            for j in range(num_cols + 1):
                plt.subplot(num_samples, num_cols + 1, i * (num_cols + 1) + j + 1)

                plt.imshow(display_list[j], cmap='gray', vmin=0, vmax=255)
                if i == 0:
                    plt.title(titles[j], fontsize=7)
                plt.axis('off')

        except Exception as err:
            print(f"Error while processing '{file_path}': {err}")

    plt.tight_layout()
    plt.show()

load_and_process_dataset(path_to_dataset)