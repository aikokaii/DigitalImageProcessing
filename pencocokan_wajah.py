import os
import sys
import argparse
import numpy as np
from PIL import Image

from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.feature_extractor import FaceFeatureExtractor
from src.pencocokan import FaceMatcher


# ============================================================
# HELPER: Cari path XML Haar Cascade otomatis (cross-platform)
# ============================================================

def find_cascade_xml(filename):
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
    raise FileNotFoundError(f"'{filename}' tidak ditemukan. Install opencv-python atau letakkan di folder ini.")


# ============================================================
# FUNGSI UTAMA: Proses 1 gambar → hasil pencocokan
# ============================================================

def proses_gambar(image_path: str, metode: str,
                  face_cascade, eye_cascade, extractor, matcher,
                  threshold_override: dict = None):
    """
    Baca gambar → deteksi wajah → crop → ekstrak fitur → cocokkan.
    Return: dict hasil pencocokan
    """
    # --- Baca gambar ---
    try:
        pil_img = Image.open(image_path).convert('RGB')
        pil_img = pil_img.resize((320, 240))
        frame   = np.array(pil_img)
    except Exception as e:
        print(f"[ERROR] Gagal membaca gambar '{image_path}': {e}")
        return None

    h_frame, w_frame, _ = frame.shape

    # --- Grayscale + equalize ---
    gray    = rgb_to_gray_manual(frame)
    gray_eq = equalize_hist(gray)

    # --- Deteksi wajah ---
    print("Mendeteksi wajah...")
    faces_raw = face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )
    faces = [(x, y, w, h) for (x, y, w, h) in faces_raw if w >= 60 and h >= 60]

    if len(faces) == 0:
        # Fallback: gunakan seluruh gambar sebagai area wajah
        print("[PERINGATAN] Wajah tidak terdeteksi, menggunakan seluruh gambar.")
        face_crop = frame
    else:
        x, y, w, h = faces[0]
        h_ext = min(int(h * 1.25), h_frame - y)  # perlebar ke bawah agar mulut tidak terpotong
        face_crop = frame[y:y + h_ext, x:x + w]
        print(f"Wajah terdeteksi: ({x},{y}) ukuran {w}x{h_ext}")

    h_crop, w_crop, _ = face_crop.shape

    # --- Deteksi mata di area crop ---
    gray_crop = rgb_to_gray_manual(face_crop)
    gray_crop_eq = equalize_hist(gray_crop)

    eyes_raw = eye_cascade.detectMultiScale(
        gray_crop_eq, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15)
    )
    filtered_eyes = [
        (ex, ey, ew, eh)
        for (ex, ey, ew, eh) in eyes_raw
        if ey < h_crop * 0.6 and ew > 10 and eh > 10
    ]
    filtered_eyes = sorted(filtered_eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

    # Skalakan koordinat mata ke 100×100
    eyes_100 = [
        (int(ex * 100 / w_crop), int(ey * 100 / h_crop),
         int(ew * 100 / w_crop), int(eh * 100 / h_crop))
        for (ex, ey, ew, eh) in filtered_eyes
    ]
    print(f"Mata terdeteksi: {len(eyes_100)}")

    # --- Resize wajah ke 100×100 ---
    pil_crop = Image.fromarray(face_crop)
    face_100 = np.array(pil_crop.resize((100, 100)))

    # --- Ekstraksi fitur ---
    print("Mengekstrak fitur...")
    hasil_ekstraksi = extractor.extract_features(face_100, eyes_100)
    fitur = hasil_ekstraksi["features"]

    # --- Pencocokan ---
    print(f"Mencocokkan ke database (metode={metode})...")
    hasil = matcher.match(fitur, metode=metode,
                          threshold_override=threshold_override)
    return hasil


# ============================================================
# VISUALISASI HASIL DI TERMINAL
# ============================================================

CONF_ICON = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "REJECTED": "🔴"}

def tampilkan_hasil(hasil, image_path: str):
    """Cetak hasil pencocokan ke terminal — hanya 1 prediksi."""
    icon = CONF_ICON.get(hasil.confidence, "❓")
    print()
    print("=" * 60)
    print("       HASIL PENCOCOKAN WAJAH (Face Recognition)")
    print("=" * 60)
    print(f"  File Input  : {os.path.basename(image_path)}")
    print(f"  Metode      : {hasil.metode.upper()}")
    print("-" * 60)

    if hasil.dikenali:
        print(f"  STATUS      : DIKENALI")
        print(f"  PREDIKSI    : {hasil.nama_prediksi.upper()}")
        print(f"  KEYAKINAN   : {icon} {hasil.confidence}")
        print(f"  Skor        : {hasil.skor_terbaik:.4f}")
    else:
        print(f"  STATUS      : TIDAK DIKENALI  {icon}")
        print(f"  Alasan      : {hasil.alasan_reject}")
        if hasil.semua_skor:
            print(f"  Kandidat    : {hasil.semua_skor[0][0]} "
                  f"(skor {hasil.semua_skor[0][1]:.4f})")

    print()
    print(f"  {'Rank':<4} {'Nama':<25} {'Skor':<10}")
    print(f"  {'─'*42}")
    for i, (nama, skor) in enumerate(hasil.semua_skor, 1):
        marker = " <-- PREDIKSI" if i == 1 and hasil.dikenali else ""
        print(f"  {i:<4} {nama:<25} {skor:<10.4f}{marker}")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pencocokan Wajah — Face Recognition Pipeline"
    )
    parser.add_argument(
        "gambar",
        help="Path ke file gambar wajah yang ingin dikenali"
    )
    parser.add_argument(
        "--metode", "-m",
        choices=["euclidean", "manhattan", "cosine"],
        default="euclidean",
        help="Metode perhitungan jarak (default: euclidean)"
    )
    parser.add_argument(
        "--database", "-d",
        default="database_fitur.json",
        help="Path ke file database fitur JSON (default: database_fitur.json)"
    )
    parser.add_argument(
        "--absolute", type=float, default=None,
        help="Override threshold absolute"
    )
    parser.add_argument(
        "--margin", type=float, default=None,
        help="Override threshold margin rank-1 vs rank-2"
    )

    args = parser.parse_args()

    # Validasi file input
    if not os.path.exists(args.gambar):
        print(f"[ERROR] File gambar '{args.gambar}' tidak ditemukan!")
        sys.exit(1)

    # Susun threshold override jika ada
    threshold_override = {}
    if args.absolute is not None:
        threshold_override["absolute"] = args.absolute
    if args.margin is not None:
        threshold_override["margin_abs"] = args.margin
    if not threshold_override:
        threshold_override = None

    # --- Load cascade ---
    print("Memuat Haar Cascade (tanpa cv2)...")
    face_cascade = ManualCascadeClassifier(find_cascade_xml('haarcascade_frontalface_default.xml'))
    eye_cascade  = ManualCascadeClassifier(find_cascade_xml('haarcascade_eye.xml'))

    # --- Inisialisasi ekstraktor & matcher ---
    extractor = FaceFeatureExtractor(target_size=(100, 100))
    matcher   = FaceMatcher(db_path=args.database)

    print(f"\nProfil terdaftar: {matcher.daftar_profil()}\n")

    # --- Proses ---
    hasil = proses_gambar(
        image_path=args.gambar,
        metode=args.metode,
        face_cascade=face_cascade,
        eye_cascade=eye_cascade,
        extractor=extractor,
        matcher=matcher,
        threshold_override=threshold_override,
    )

    if hasil:
        tampilkan_hasil(hasil, args.gambar)


if __name__ == "__main__":
    main()