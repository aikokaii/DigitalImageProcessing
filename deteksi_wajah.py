import numpy as np
import os
import sys
from PIL import Image

from src.face_utils import (
    gaussian_blur_manual,
    sobel_edge_detection,
    skin_segmentation_manual
)

from src.manual_cascade import ManualCascadeClassifier

from src.image_utils import (
    equalize_hist,
    rgb_to_gray_manual
)

# =========================
# LOAD CASCADE
# =========================
cv2_data_dir = r"C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\cv2\data"

if not os.path.exists(cv2_data_dir):

    import site

    for sp in site.getsitepackages() + [site.getusersitepackages()]:

        potential_path = os.path.join(sp, 'cv2', 'data')

        if os.path.exists(potential_path):
            cv2_data_dir = potential_path
            break

print("Loading manual Haar Cascade for Face from XML... (this may take a moment)")

face_cascade = ManualCascadeClassifier(
    os.path.join(
        cv2_data_dir,
        'haarcascade_frontalface_default.xml'
    )
)

print("Loading manual Haar Cascade for Eyes from XML...")

eye_cascade = ManualCascadeClassifier(
    os.path.join(
        cv2_data_dir,
        'haarcascade_eye.xml'
    )
)

# =========================
# FOLDER INPUT / OUTPUT
# =========================
input_folder = "webcam"

output_skin = "hasil_skin"
output_edge = "hasil_edge"
output_detection = "hasil_deteksi"

os.makedirs(output_skin, exist_ok=True)
os.makedirs(output_edge, exist_ok=True)
os.makedirs(output_detection, exist_ok=True)

# =========================
# VALIDASI FOLDER INPUT
# =========================
if not os.path.exists(input_folder):

    print(f"Folder input tidak ditemukan: {input_folder}")
    sys.exit()

# =========================
# AMBIL SEMUA GAMBAR
# =========================
image_files = [

    f for f in os.listdir(input_folder)

    if (
        f.endswith(".png")
        or
        f.endswith(".pgm")
    )
]

# =========================
# VARIABEL EVALUASI
# =========================
total_images = 0

face_detected_count = 0
face_failed_count = 0

eye_detected_count = 0
expected_eye_count = 0

# =========================
# PROSES GAMBAR
# =========================
for image_name in image_files:

    total_images += 1

    print(f"\nMemproses: {image_name}")

    image_path = os.path.join(
        input_folder,
        image_name
    )

    # =========================
    # LOAD IMAGE
    # =========================
    try:

        pil_img = Image.open(image_path).convert('RGB')

        pil_img = pil_img.resize((320, 240))

        frame = np.array(pil_img)

    except Exception as e:

        print(f"Gagal memuat {image_name}: {e}")
        continue

    # =========================
    # PREPROCESSING
    # =========================
    gray = rgb_to_gray_manual(frame)

    gray = equalize_hist(gray)

    print(
        f"-> Detecting faces using from-scratch cascade on {image_name}..."
    )

    # =========================
    # FACE DETECTION
    # =========================
    raw_faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=3,

        minSize=(30, 30)
    )

    # =========================
    # FILTER UKURAN WAJAH
    # =========================
    faces = [

        (x, y, w, h)

        for (x, y, w, h) in raw_faces

        if w >= 60 and h >= 60
    ]

    # =========================
    # EVALUASI DETEKSI WAJAH
    # =========================
    face_detected = len(faces) > 0

    if face_detected:

        face_detected_count += 1

        print(
            f"[INFO] Wajah berhasil terdeteksi pada {image_name}"
        )

    else:

        face_failed_count += 1

        print(
            f"[INFO] Wajah gagal terdeteksi pada {image_name}"
        )

    # =========================
    # JIKA TIDAK ADA WAJAH -> BUANG
    # =========================
    if not face_detected:

        print(
            f"Wajah tidak terdeteksi pada {image_name}, membuang gambar (tidak disimpan)."
        )

        continue

    # =========================
    # DEFAULT OUTPUT
    # =========================
    final_crop = frame

    skin_mask = None

    edges = None

    # =========================
    # LOOP FACE
    # =========================
    for (x, y, w, h) in faces:

        # =========================
        # PERLEBAR AREA WAJAH
        # =========================
        h_new = int(h * 1.25)

        if y + h_new > frame.shape[0]:

            h_new = frame.shape[0] - y

        # =========================
        # ROI
        # =========================
        roi_gray = gray[
            y:y+h_new,
            x:x+w
        ]

        roi_color = frame[
            y:y+h_new,
            x:x+w
        ]

        # =========================
        # SAVE FACE CROP
        # =========================
        final_crop = roi_color

        # =========================
        # BLUR
        # =========================
        blurred_crop = gaussian_blur_manual(
            roi_gray
        )

        # =========================
        # RGB -> HSV
        # =========================
        pil_img_crop = Image.fromarray(
            roi_color
        )

        pil_hsv_crop = np.array(
            pil_img_crop.convert('HSV')
        )

        hsv_crop = np.zeros_like(
            pil_hsv_crop
        )

        hsv_crop[:, :, 0] = (

            pil_hsv_crop[:, :, 0]
            .astype(np.float32)

            * 179 / 255

        ).astype(np.uint8)

        hsv_crop[:, :, 1] = pil_hsv_crop[:, :, 1]

        hsv_crop[:, :, 2] = pil_hsv_crop[:, :, 2]

        # =========================
        # SKIN SEGMENTATION
        # =========================
        skin_mask = skin_segmentation_manual(
            hsv_crop
        )

        # =========================
        # EDGE DETECTION
        # =========================
        edges = sobel_edge_detection(
            blurred_crop
        )

        # =========================
        # EYE DETECTION
        # =========================
        h = h_new

        eyes = eye_cascade.detectMultiScale(

            roi_gray,

            scaleFactor=1.05,

            minNeighbors=9,

            minSize=(20, 20)
        )

        # =========================
        # FILTER MATA
        # =========================
        filtered_eyes = [

            (ex, ey, ew, eh)

            for (ex, ey, ew, eh) in eyes

            if (
                ey < h * 0.5
                and
                ew > 15
                and
                eh > 15
            )
        ]

        # =========================
        # AMBIL 2 MATA TERBAIK
        # =========================
        filtered_eyes = sorted(

            filtered_eyes,

            key=lambda e: e[2] * e[3],

            reverse=True

        )[:2]

        filtered_eyes = sorted(
            filtered_eyes,
            key=lambda e: e[0]
        )

        # =========================
        # EVALUASI DETEKSI MATA
        # HANYA JIKA WAJAH VALID
        # =========================
        if face_detected:

            expected_eye_count += 2

            eye_detected_count += len(
                filtered_eyes
            )

    # =========================
    # OUTPUT FOLDER
    # =========================
    if "datatest" in image_name.lower():

        target_folder = "datatest"

        out_skin = "datatest_skin"

        out_edge = "datatest_edge"

    else:

        target_folder = output_detection

        out_skin = output_skin

        out_edge = output_edge

    os.makedirs(target_folder, exist_ok=True)

    os.makedirs(out_skin, exist_ok=True)

    os.makedirs(out_edge, exist_ok=True)

    filename = os.path.splitext(
        image_name
    )[0]

    # =========================
    # SAVE SKIN
    # =========================
    if skin_mask is not None and edges is not None:

        Image.fromarray(

            skin_mask,

            mode='L'

        ).save(

            os.path.join(
                out_skin,
                filename + "_skin.png"
            )
        )

        # =========================
        # SAVE EDGE
        # =========================
        Image.fromarray(

            edges,

            mode='L'

        ).save(

            os.path.join(
                out_edge,
                filename + "_edge.png"
            )
        )

    # =========================
    # SAVE FACE CROP
    # =========================
    Image.fromarray(

        final_crop,

        mode='RGB'

    ).save(

        os.path.join(

            target_folder,

            f"{filename}_face detected.png"
        )
    )

    print(
        f"Selesai: {image_name} (Tersimpan di folder '{target_folder}')"
    )

# =========================
# HASIL EVALUASI AKURASI
# =========================
print("\n==============================")

print("HASIL EVALUASI SISTEM")

print("==============================")

print(f"Total gambar               : {total_images}")

print(f"Wajah berhasil terdeteksi : {face_detected_count}")

print(f"Wajah gagal terdeteksi    : {face_failed_count}")

# =========================
# AKURASI WAJAH
# =========================
if total_images > 0:

    face_accuracy = (

        face_detected_count
        /
        total_images

    ) * 100

else:

    face_accuracy = 0

print(
    f"Akurasi deteksi wajah     : {face_accuracy:.2f}%"
)

# =========================
# AKURASI MATA
# =========================
if expected_eye_count > 0:

    eye_accuracy = (

        eye_detected_count
        /
        expected_eye_count

    ) * 100

else:

    eye_accuracy = 0

print(
    f"Total mata terdeteksi     : {eye_detected_count}"
)

print(
    f"Total mata seharusnya     : {expected_eye_count}"
)

print(
    f"Akurasi deteksi mata      : {eye_accuracy:.2f}%"
)

print("==============================")

print("Semua gambar berhasil diproses")