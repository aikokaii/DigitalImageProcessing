import numpy as np
import os
import cv2
import csv
import base64
from datetime import datetime

# =====================================================
# HILBERT CURVE COMPRESSOR
# =====================================================

class HilbertCurveCompressor:

    def __init__(self, n):

        self.n = n
        self.size = 1 << n

    def _hilbert_curve(self, n):

        if n == 0:
            return np.array([[0, 0]])

        prev = self._hilbert_curve(n - 1)

        m = 1 << (n - 1)

        q1 = np.column_stack((
            prev[:, 1],
            prev[:, 0]
        ))

        q2 = np.column_stack((
            prev[:, 0],
            prev[:, 1] + m
        ))

        q3 = np.column_stack((
            prev[:, 0] + m,
            prev[:, 1] + m
        ))

        q4 = np.column_stack((
            2 * m - 1 - prev[:, 1],
            m - 1 - prev[:, 0]
        ))

        return np.vstack((q1, q2, q3, q4))

    def get_curve_indices(self):

        return self._hilbert_curve(self.n)

    def image_to_curve(self, gray_image):

        assert gray_image.shape == (
            self.size,
            self.size
        )

        curve = self.get_curve_indices()

        return np.array([
            gray_image[y, x]
            for y, x in curve
        ], dtype=np.uint8)

    def curve_to_image(self, curve_1d):

        img = np.zeros(
            (self.size, self.size),
            dtype=np.uint8
        )

        curve = self.get_curve_indices()

        for idx, (y, x) in enumerate(curve):

            img[y, x] = curve_1d[idx]

        return img


# =====================================================
# KOMPRESI
# =====================================================

def compress_grayscale(
    gray_img,
    hilbert_compressor
):

    curve_data = hilbert_compressor.image_to_curve(
        gray_img
    )

    diff = np.diff(
        curve_data.astype(np.int16),
        prepend=curve_data[0]
    )

    compressed = []

    run_val = diff[0]
    run_len = 1

    for val in diff[1:]:

        if val == run_val and run_len < 255:

            run_len += 1

        else:

            compressed.extend([
                run_val,
                run_len
            ])

            run_val = val
            run_len = 1

    compressed.extend([
        run_val,
        run_len
    ])

    return np.array(
        compressed,
        dtype=np.int16
    ).tobytes()


# =====================================================
# DEKOMPRESI
# =====================================================

def decompress_grayscale(
    compressed_bytes,
    hilbert_compressor,
    original_length
):

    diff_rle = np.frombuffer(
        compressed_bytes,
        dtype=np.int16
    )

    diffs = []

    for i in range(0, len(diff_rle), 2):

        val = diff_rle[i]

        length = diff_rle[i + 1]

        diffs.extend([val] * length)

    diffs = np.array(
        diffs[:original_length],
        dtype=np.int16
    )

    curve_reconstructed = np.cumsum(
        diffs
    ).astype(np.uint8)

    return hilbert_compressor.curve_to_image(
        curve_reconstructed
    )


# =====================================================
# CSV DATABASE
# =====================================================

csv_database = "dataset.csv"

if not os.path.exists(csv_database):

    with open(
        csv_database,
        mode='w',
        newline='',
        encoding='utf-8'
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "face_id",
            "filename",
            "timestamp",
            "image_size",
            "hilbert_order",
            "compressed_data"

        ])


# =====================================================
# FOLDER WEBCAM
# =====================================================

input_folder = "webcam"

if not os.path.exists(input_folder):

    print("Folder webcam tidak ditemukan")
    exit()


# =====================================================
# HILBERT SETTING
# =====================================================

order = 8

hilbert_comp = HilbertCurveCompressor(order)


# =====================================================
# AMBIL FILE GAMBAR
# =====================================================

image_files = []

for file in os.listdir(input_folder):

    if (
        file.endswith(".png")
        or
        file.endswith(".pgm")
    ):

        image_files.append(file)


# =====================================================
# PROSES SEMUA GAMBAR
# =====================================================

for index, image_name in enumerate(image_files):

    print(f"Memproses: {image_name}")

    image_path = os.path.join(
        input_folder,
        image_name
    )

    # =================================================
    # LOAD IMAGE
    # =================================================

    if image_name.endswith(".pgm"):

        gray_img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

    else:

        img = cv2.imread(image_path)

        if img is None:
            continue

        gray_img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    # =================================================
    # RESIZE
    # =================================================

    gray_img = cv2.resize(
        gray_img,
        (256, 256)
    )

    # =================================================
    # KOMPRESI HILBERT
    # =================================================

    compressed_bytes = compress_grayscale(
        gray_img,
        hilbert_comp
    )

    # =================================================
    # BYTE → BASE64
    # =================================================

    compressed_base64 = base64.b64encode(
        compressed_bytes
    ).decode('utf-8')

    # =================================================
    # TIMESTAMP
    # =================================================

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # =================================================
    # FACE ID
    # =================================================

    filename_only = os.path.splitext(
        image_name
    )[0]

    face_id = (
        f"FACE_{index+1}_"
        f"{filename_only}"
    )

    # =================================================
    # SIMPAN KE CSV
    # =================================================

    with open(
        csv_database,
        mode='a',
        newline='',
        encoding='utf-8'
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            face_id,
            image_name,
            timestamp,
            gray_img.size,
            order,
            compressed_base64

        ])

    print(
        f"Berhasil disimpan ke CSV: {face_id}"
    )


# =====================================================
# FINISH
# =====================================================

print("Semua gambar webcam berhasil diproses")