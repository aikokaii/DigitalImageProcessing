import cv2
import numpy as np
import os

# =========================
# LOAD HAAR CASCADE
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_eye.xml'
)

# =========================
# FOLDER INPUT
# =========================

input_folder = "webcam"

# =========================
# FOLDER OUTPUT
# =========================

output_skin = "hasil_skin"
output_edge = "hasil_edge"
output_detection = "hasil_deteksi"

os.makedirs(output_skin, exist_ok=True)
os.makedirs(output_edge, exist_ok=True)
os.makedirs(output_detection, exist_ok=True)

# =========================
# GAUSSIAN BLUR MANUAL
# =========================

def gaussian_blur_manual(image):

    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32)

    kernel = kernel / 16.0

    height, width = image.shape

    output = np.zeros_like(image)

    padded = np.pad(
        image,
        ((1, 1), (1, 1)),
        mode='constant'
    )

    for y in range(height):

        for x in range(width):

            region = padded[
                y:y+3,
                x:x+3
            ]

            value = np.sum(region * kernel)

            output[y, x] = value

    return output

# =========================
# SOBEL MANUAL
# =========================

def sobel_edge_detection(image):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])

    height, width = image.shape

    output = np.zeros_like(image)

    padded = np.pad(
        image,
        ((1, 1), (1, 1)),
        mode='constant'
    )

    for y in range(height):

        for x in range(width):

            region = padded[
                y:y+3,
                x:x+3
            ]

            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)

            magnitude = np.sqrt(gx**2 + gy**2)

            magnitude = min(255, magnitude)

            output[y, x] = magnitude

    return output.astype(np.uint8)

# =========================
# SKIN SEGMENTATION MANUAL
# =========================

def skin_segmentation_manual(hsv):

    height, width, _ = hsv.shape

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for y in range(height):

        for x in range(width):

            h = hsv[y, x, 0]
            s = hsv[y, x, 1]
            v = hsv[y, x, 2]

            if (
                0 <= h <= 20 and
                30 <= s <= 150 and
                60 <= v <= 255
            ):

                mask[y, x] = 255

    return mask

# =========================
# AMBIL SEMUA GAMBAR
# =========================

image_files = []

for file in os.listdir(input_folder):

    if file.endswith(".png") or file.endswith(".pgm"):

        image_files.append(file)

# =========================
# PROSES GAMBAR
# =========================

for image_name in image_files:

    print(f"Memproses: {image_name}")

    image_path = os.path.join(
        input_folder,
        image_name
    )

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    # Resize lebih kecil agar cepat
    frame = cv2.resize(
        frame,
        (320, 240)
    )

    # =========================
    # PREPROCESSING
    # =========================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    # =========================
    # GAUSSIAN BLUR MANUAL
    # =========================

    blurred = gaussian_blur_manual(gray)

    # =========================
    # HSV
    # =========================

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # =========================
    # SKIN SEGMENTATION MANUAL
    # =========================

    skin_mask = skin_segmentation_manual(hsv)

    # =========================
    # EDGE DETECTION MANUAL
    # =========================

    edges = sobel_edge_detection(blurred)

    # =========================
    # FACE DETECTION
    # =========================

    faces = face_cascade.detectMultiScale(
        blurred,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # =========================
    # DETEKSI WAJAH
    # =========================

    for (x, y, w, h) in faces:

        # ROI wajah
        roi_gray = blurred[
            y:y+h,
            x:x+w
        ]

        roi_color = frame[
            y:y+h,
            x:x+w
        ]

        # =========================
        # TEMPLATE MATCHING
        # =========================
        # Bounding box wajah dipakai
        # sebagai area template
        # agar tidak salah sasaran
        # =========================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        # =========================
        # LINGKARAN WAJAH
        # =========================

        center_face = (
            x + w // 2,
            y + h // 2
        )

        radius_face = int(w * 0.52)

        cv2.circle(
            frame,
            center_face,
            radius_face,
            (255, 0, 255),
            3
        )

        # =========================
        # DETEKSI MATA
        # =========================

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=9,
            minSize=(20, 20)
        )

        filtered_eyes = []

        for (ex, ey, ew, eh) in eyes:

            # Mata hanya bagian atas wajah
            if ey < h * 0.5:

                # Hindari objek kecil
                if ew > 15 and eh > 15:

                    filtered_eyes.append(
                        (ex, ey, ew, eh)
                    )

        # =========================
        # PILIH 2 MATA TERBESAR
        # =========================

        filtered_eyes = sorted(
            filtered_eyes,
            key=lambda e: e[2] * e[3],
            reverse=True
        )[:2]

        # Urut kiri ke kanan
        filtered_eyes = sorted(
            filtered_eyes,
            key=lambda e: e[0]
        )

        # =========================
        # GAMBAR MATA
        # =========================

        for (ex, ey, ew, eh) in filtered_eyes:

            center_eye = (
                ex + ew // 2,
                ey + eh // 2
            )

            radius_eye = int(ew * 0.45)

            cv2.circle(
                roi_color,
                center_eye,
                radius_eye,
                (255, 0, 0),
                3
            )

    # =========================
    # NAMA FILE
    # =========================

    filename = os.path.splitext(
        image_name
    )[0]

    # =========================
    # SIMPAN HASIL
    # =========================

    cv2.imwrite(
        os.path.join(
            output_skin,
            filename + "_skin.png"
        ),
        skin_mask
    )

    cv2.imwrite(
        os.path.join(
            output_edge,
            filename + "_edge.png"
        ),
        edges
    )

    cv2.imwrite(
        os.path.join(
            output_detection,
            filename + "_detected.png"
        ),
        frame
    )

    print(f"Selesai: {image_name}")

# =========================
# SELESAI
# =========================

cv2.destroyAllWindows()

print("Semua gambar berhasil diproses")