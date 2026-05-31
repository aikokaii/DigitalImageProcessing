import os
import numpy as np
from PIL import Image

from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.feature_extractor import FaceFeatureExtractor
from src.database import FaceDatabase

def find_cascade_xml(filename):
    """Cari file XML cascade tanpa hardcode path."""
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(local):
        return local
    try:
        import site
        all_site = site.getsitepackages()
        try:
            all_site += [site.getusersitepackages()]
        except Exception:
            pass
        for sp in all_site:
            p = os.path.join(sp, 'cv2', 'data', filename)
            if os.path.exists(p):
                return p
    except Exception:
        pass
    for p in [f"/usr/share/opencv4/haarcascades/{filename}",
              f"/usr/local/share/opencv4/haarcascades/{filename}"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"'{filename}' tidak ditemukan.\n"
        "Pastikan OpenCV terinstall: pip install opencv-python\n"
        "ATAU letakkan file XML di folder yang sama dengan script ini."
    )

def main():
    print("Memuat model Deteksi Mata (Manual Cascade, tanpa cv2)...")
    eye_xml_path = find_cascade_xml('haarcascade_eye.xml')
    print(f"  -> XML ditemukan: {eye_xml_path}")
    eye_cascade = ManualCascadeClassifier(eye_xml_path)

    extractor = FaceFeatureExtractor(target_size=(100, 100))

    db_path = "database_fitur.json"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"  -> Database lama '{db_path}' dihapus, akan dibuat ulang.")
    db = FaceDatabase(db_path)

    input_folder = "hasil_deteksi"
    if not os.path.exists(input_folder):
        print(f"[ERROR] Folder '{input_folder}' tidak ditemukan!")
        print("Jalankan deteksi_wajah.py terlebih dahulu.")
        return

    face_files = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith('face detected.png')
    ])

    if not face_files:
        print(f"[ERROR] Tidak ada file '*_face detected.png' di '{input_folder}'.")
        print("Jalankan deteksi_wajah.py terlebih dahulu.")
        return

    print(f"\nDitemukan {len(face_files)} gambar wajah di '{input_folder}'.")
    print("Memulai ekstraksi fitur dataset...\n")

    berhasil = 0
    gagal    = 0

    for i, image_name in enumerate(face_files, 1):
        nama_orang = (
            image_name
            .replace('_face detected.png', '')
            .split('(')[0]
            .strip()
            .lower()
        )
        image_path = os.path.join(input_folder, image_name)

        print(f"[{i}/{len(face_files)}] Memproses: {image_name}")

        try:
            pil_img      = Image.open(image_path).convert('RGB')
            face_crop_rgb = np.array(pil_img)
            h_orig, w_orig, _ = face_crop_rgb.shape
        except Exception as e:
            print(f"  [SKIP] Gagal memuat gambar: {e}")
            gagal += 1
            continue

        # --- A. Deteksi mata pada crop asli ---
        gray    = rgb_to_gray_manual(face_crop_rgb)
        gray_eq = equalize_hist(gray)

        eyes_raw = eye_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15)
        )

        filtered_eyes = [
            (ex, ey, ew, eh)
            for (ex, ey, ew, eh) in eyes_raw
            if ey < h_orig * 0.6 and ew > 10 and eh > 10
        ]
        filtered_eyes = sorted(
            filtered_eyes, key=lambda e: e[2] * e[3], reverse=True
        )[:2]

        # --- B. Skalakan koordinat mata ke 100×100 ---
        eyes_100 = [
            (int(ex * 100 / w_orig), int(ey * 100 / h_orig),
             int(ew * 100 / w_orig), int(eh * 100 / h_orig))
            for (ex, ey, ew, eh) in filtered_eyes
        ]

        # --- C. Resize wajah ke 100×100 ---
        face_100 = np.array(pil_img.resize((100, 100)))

        # --- D. Ekstraksi Fitur ---
        try:
            hasil   = extractor.extract_features(face_100, eyes_100)
            fitur   = hasil["features"]
            fitur["file"] = image_name
            masks   = hasil["masks"]
        except Exception as e:
            print(f"  [SKIP] Gagal ekstraksi fitur: {e}")
            gagal += 1
            continue


        # --- E. Simpan mask untuk visualisasi ---
        # NOTE: Dinonaktifkan agar tidak mengotori folder hasil_deteksi
        # base = image_name.replace('face detected.png', '')
        # Image.fromarray(masks["skin"],    mode='L').save(
        #     os.path.join(input_folder, base + 'skin_crop.png'))
        # Image.fromarray(masks["edge"],    mode='L').save(
        #     os.path.join(input_folder, base + 'edge_crop.png'))
        # Image.fromarray(masks["contour"], mode='L').save(
        #     os.path.join(input_folder, base + 'contour_crop.png'))

        # --- F. Simpan ke database ---
        db.add_profile(nama_orang, fitur)
        print(f"  -> Profil '{nama_orang}' disimpan (mata: {len(eyes_100)})")
        berhasil += 1

    # --- 4. Tulis database ---
    print(f"\n{'='*45}")
    print(f"Ekstraksi selesai: {berhasil} berhasil, {gagal} gagal.")
    print(f"Profil terdaftar : {list(db.data.keys())}")
    db.save()
    print(f"{'='*45}")


if __name__ == "__main__":
    main()