import os
import sys
import json
import numpy as np
from PIL import Image

from src.feature_extractor import FaceFeatureExtractor
from src.database import FaceDatabase
from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.image_utils import rgb_to_gray_manual, equalize_hist

def find_cascade_xml(filename):
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(local):
        return local
    try:
        import site
        all_site = site.getsitepackages()
        all_site += [site.getusersitepackages()]
        for sp in all_site:
            p = os.path.join(sp, 'cv2', 'data', filename)
            if os.path.exists(p):
                return p
    except:
        pass
    raise FileNotFoundError(f"'{filename}' tidak ditemukan.")

def main():
    folder_wajah = "datatest"
    db_file = "database_uji.json"

    if not os.path.exists(folder_wajah):
        print(f"[ERROR] Folder '{folder_wajah}' tidak ditemukan!")
        sys.exit(1)

    print("Memuat model Deteksi Mata (Manual Cascade, tanpa cv2)...")
    try:
        eye_cascade_path = find_cascade_xml('haarcascade_eye.xml')
        eye_cascade = ManualCascadeClassifier(eye_cascade_path)
        print(f"  -> XML ditemukan: {eye_cascade_path}\n")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    extractor = FaceFeatureExtractor(target_size=(100, 100))
    # Buat instance db baru atau timpa yang lama
    if os.path.exists(db_file):
        os.remove(db_file)
    db = FaceDatabase(db_file)

    face_files = sorted(os.listdir(folder_wajah))
    valid_files = [f for f in face_files if f.endswith('_face detected.png')]

    if len(valid_files) == 0:
        print(f"[INFO] Tidak ada file '*_face detected.png' di '{folder_wajah}'.")
        sys.exit(0)

    print("Memulai ekstraksi fitur dataset uji (Testing)...")
    for i, image_name in enumerate(valid_files):
        print(f"\n[{i+1}/{len(valid_files)}] Memproses uji: {image_name}")
        image_path = os.path.join(folder_wajah, image_name)

        # Nama orang
        nama_orang = (
            image_name
            .replace('_face detected.png', '')
            .split('(')[0]
            .strip()
            .lower()
        )

        try:
            pil_img = Image.open(image_path).convert('RGB')
            face_crop = np.array(pil_img)
        except Exception as e:
            print(f"  [ERROR] Gagal memuat {image_name}: {e}")
            continue

        h_crop, w_crop, _ = face_crop.shape
        gray_crop = rgb_to_gray_manual(face_crop)
        gray_crop_eq = equalize_hist(gray_crop)

        eyes_raw = eye_cascade.detectMultiScale(
            gray_crop_eq, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15)
        )
        filtered_eyes = [
            (ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes_raw
            if ey < h_crop * 0.6 and ew > 10 and eh > 10
        ]
        filtered_eyes = sorted(filtered_eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

        eyes_100 = [
            (int(ex * 100 / w_crop), int(ey * 100 / h_crop),
             int(ew * 100 / w_crop), int(eh * 100 / h_crop))
            for (ex, ey, ew, eh) in filtered_eyes
        ]

        face_100 = np.array(pil_img.resize((100, 100)))

        hasil_ekstraksi = extractor.extract_features(face_100, eyes_100)
        fitur = hasil_ekstraksi["features"]
        fitur["file"] = image_name

        db.add_profile(nama_orang, fitur)
        print(f"  -> Profil uji '{nama_orang}' disimpan ke {db_file} (mata terdeteksi: {len(eyes_100)})")

    db.save()
    print(f"\n[SELESAI] Data fitur uji (datatest) berhasil disimpan ke {db_file}")

if __name__ == "__main__":
    main()
