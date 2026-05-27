import cv2
import numpy as np

# =========================
# LOAD CASCADE CLASSIFIER
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak ditemukan")
    exit()

print("Tekan Q untuk keluar")

while True:

    # =========================
    # AMBIL FRAME
    # =========================
    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Resize agar ringan
    frame = cv2.resize(frame, (640, 480))

    # =========================
    # PREPROCESSING
    # =========================

    # Convert ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perbaiki kontras
    gray = cv2.equalizeHist(gray)

    # =========================
    # SKIN COLOR SEGMENTATION
    # =========================

    # Convert ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range warna kulit
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    # Membuat mask kulit
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Mengurangi noise
    kernel = np.ones((5, 5), np.uint8)

    skin_mask = cv2.erode(
        skin_mask,
        kernel,
        iterations=1
    )

    skin_mask = cv2.dilate(
        skin_mask,
        kernel,
        iterations=2
    )

    # Blur supaya lebih halus
    skin_mask = cv2.GaussianBlur(
        skin_mask,
        (5, 5),
        0
    )

    # =========================
    # EDGE DETECTION
    # =========================

    edges = cv2.Canny(gray, 100, 200)

    # =========================
    # HAAR CASCADE FACE DETECTION
    # =========================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # =========================
    # DETEKSI WAJAH
    # =========================

    for (x, y, w, h) in faces:

        # =========================
        # TEMPLATE MATCHING SEDERHANA
        # =========================

        face_roi = gray[y:y+h, x:x+w]

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
        # ROI WAJAH
        # =========================

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # =========================
        # DETEKSI MATA
        # =========================

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=9,
            minSize=(25, 25)
        )

        filtered_eyes = []

        # =========================
        # FILTER MATA
        # =========================

        for (ex, ey, ew, eh) in eyes:

            # Mata harus di bagian atas wajah
            if ey < h * 0.5:

                # Hindari objek kecil
                if ew > 30 and eh > 30:

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
    # TAMPILKAN HASIL
    # =========================

    cv2.imshow(
        "Face Detection",
        frame
    )

    cv2.imshow(
        "Skin Segmentation",
        skin_mask
    )

    cv2.imshow(
        "Edge Detection",
        edges
    )

    # =========================
    # EXIT
    # =========================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()
cv2.destroyAllWindows()