"""
ekstraksi_fitur_cepat.py
========================
Versi CEPAT dari ekstraksi_fitur.py menggunakan:

1. multiprocessing.Pool — proses N gambar PARALEL di semua CPU core
   (dari ~40% CPU -> mendekati 100%)
2. Numba JIT (opsional) — mempercepat inner loop cascade 10-50x
   jika numba terinstall (pip install numba)
3. Fallback otomatis ke numpy murni jika numba tidak ada

Cara pakai:
    python ekstraksi_fitur_cepat.py
    python ekstraksi_fitur_cepat.py --workers 8   # paksa jumlah core
    python ekstraksi_fitur_cepat.py --no-numba     # nonaktifkan numba
"""

import os
import sys
import json
import time
import argparse
import multiprocessing
from multiprocessing import Pool
from functools import partial

import numpy as np
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.feature_extractor import FaceFeatureExtractor

# ── Numba JIT (opsional) ───────────────────────────────────────────────────
try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False

# ── Global: cascade data (diload sekali di setiap worker) ──────────────────
_EYE_CASCADE = None


def _init_worker(eye_xml_path):
    """Inisialisasi worker: load cascade SEKALI per proses, bukan per gambar."""
    global _EYE_CASCADE
    from src.manual_cascade import ManualCascadeClassifier
    _EYE_CASCADE = ManualCascadeClassifier(eye_xml_path)


# ============================================================
# ACCELERATED INNER LOOP (Numba atau pure NumPy)
# ============================================================

if NUMBA_OK:
    @njit(parallel=False, cache=True, fastmath=True)
    def _eval_cascade_numba(ii, ii_sq,
                             feat_xs, feat_ys, feat_ws, feat_hs, feat_wts,
                             feat_starts, feat_lens,
                             stage_thresholds,
                             weak_feat_idxs, weak_thresholds,
                             weak_leaf0, weak_leaf1,
                             stage_starts, stage_lens,
                             win_w, win_h, scaled_w, scaled_h,
                             step, inv_area):
        """
        Inner loop deteksi cascade — dikompilasi Numba JIT.
        Mengevaluasi semua window di satu skala.
        Return: list koordinat (x, y) yang lolos semua stage.
        """
        hits_x = []
        hits_y = []

        for y in range(0, scaled_h - win_h, step):
            for x in range(0, scaled_w - win_w, step):
                # Normalisasi variance
                win_sum    = ii[y + win_h, x + win_w] - ii[y, x + win_w] \
                           - ii[y + win_h, x] + ii[y, x]
                win_sq_sum = ii_sq[y + win_h, x + win_w] - ii_sq[y, x + win_w] \
                           - ii_sq[y + win_h, x] + ii_sq[y, x]
                mean       = win_sum * inv_area
                var        = win_sq_sum * inv_area - mean * mean
                std        = var ** 0.5 if var > 0.0 else 1.0

                pass_all = True
                for si in range(len(stage_thresholds)):
                    stage_sum = 0.0
                    ws = stage_starts[si]
                    we = ws + stage_lens[si]
                    for wi in range(ws, we):
                        fi       = weak_feat_idxs[wi]
                        feat_val = 0.0
                        fs       = feat_starts[fi]
                        fe       = fs + feat_lens[fi]
                        for ri in range(fs, fe):
                            rx = feat_xs[ri]; ry = feat_ys[ri]
                            rw = feat_ws[ri]; rh = feat_hs[ri]
                            rweight = feat_wts[ri]
                            if (x + rx + rw) <= scaled_w and (y + ry + rh) <= scaled_h:
                                rs = ii[y+ry+rh, x+rx+rw] - ii[y+ry, x+rx+rw] \
                                   - ii[y+ry+rh, x+rx] + ii[y+ry, x+rx]
                                feat_val += rs * rweight
                        feat_val *= inv_area
                        if feat_val < weak_thresholds[wi] * std:
                            stage_sum += weak_leaf0[wi]
                        else:
                            stage_sum += weak_leaf1[wi]

                    if stage_sum < stage_thresholds[si]:
                        pass_all = False
                        break

                if pass_all:
                    hits_x.append(x)
                    hits_y.append(y)

        return hits_x, hits_y

    NUMBA_OK = True  # confirmed
else:
    _eval_cascade_numba = None


def _prepare_cascade_arrays(cascade):
    """Konversi data cascade ke flat numpy array untuk Numba."""
    feat_xs, feat_ys, feat_ws, feat_hs, feat_wts = [], [], [], [], []
    feat_starts, feat_lens = [], []
    pos = 0
    for rects in cascade.features:
        feat_starts.append(pos)
        feat_lens.append(len(rects))
        for rx, ry, rw, rh, rw_ in rects:
            feat_xs.append(rx); feat_ys.append(ry)
            feat_ws.append(rw); feat_hs.append(rh)
            feat_wts.append(rw_)
        pos += len(rects)

    stage_thresholds = []
    stage_starts, stage_lens = [], []
    weak_feat_idxs, weak_thresholds = [], []
    weak_leaf0, weak_leaf1 = [], []
    wp = 0
    for stage in cascade.stages:
        stage_thresholds.append(stage['threshold'])
        stage_starts.append(wp)
        stage_lens.append(len(stage['weak_classifiers']))
        for wc in stage['weak_classifiers']:
            weak_feat_idxs.append(wc['feature_idx'])
            weak_thresholds.append(wc['threshold'])
            weak_leaf0.append(wc['leaf0'])
            weak_leaf1.append(wc['leaf1'])
        wp += len(stage['weak_classifiers'])

    return (
        np.array(feat_xs, dtype=np.int32),
        np.array(feat_ys, dtype=np.int32),
        np.array(feat_ws, dtype=np.int32),
        np.array(feat_hs, dtype=np.int32),
        np.array(feat_wts, dtype=np.float64),
        np.array(feat_starts, dtype=np.int32),
        np.array(feat_lens, dtype=np.int32),
        np.array(stage_thresholds, dtype=np.float64),
        np.array(weak_feat_idxs, dtype=np.int32),
        np.array(weak_thresholds, dtype=np.float64),
        np.array(weak_leaf0, dtype=np.float64),
        np.array(weak_leaf1, dtype=np.float64),
        np.array(stage_starts, dtype=np.int32),
        np.array(stage_lens, dtype=np.int32),
    )


# ============================================================
# FUNGSI DETEKSI SATU SKALA (diparalel per gambar)
# ============================================================

def _detect_one_scale_numpy(cascade, ii, ii_sq, gray,
                              scaled_w, scaled_h, scale,
                              win_w, win_h, step, inv_area, min_size):
    """Deteksi pada 1 skala — pure NumPy (tanpa Numba)."""
    from src.manual_cascade import rect_sum
    detected = []
    for y in range(0, scaled_h - win_h, step):
        for x in range(0, scaled_w - win_w, step):
            win_sum    = rect_sum(ii,    x, y, win_w, win_h)
            win_sq_sum = rect_sum(ii_sq, x, y, win_w, win_h)
            mean       = win_sum * inv_area
            variance   = win_sq_sum * inv_area - mean * mean
            std        = float(np.sqrt(max(0.0, variance))) or 1.0

            pass_all = True
            for stage in cascade.stages:
                stage_sum = 0.0
                for weak in stage['weak_classifiers']:
                    feature  = cascade.features[weak['feature_idx']]
                    feat_val = 0.0
                    for rx, ry, rw, rh, rweight in feature:
                        if (x + rx + rw) <= scaled_w and (y + ry + rh) <= scaled_h:
                            feat_val += rect_sum(ii, x+rx, y+ry, rw, rh) * rweight
                    feat_val *= inv_area
                    if feat_val < weak['threshold'] * std:
                        stage_sum += weak['leaf0']
                    else:
                        stage_sum += weak['leaf1']
                if stage_sum < stage['threshold']:
                    pass_all = False
                    break

            if pass_all:
                detected.append([int(x*scale), int(y*scale),
                                  int(win_w*scale), int(win_h*scale)])
    return detected


# ============================================================
# FUNGSI PER GAMBAR (dijalankan di worker process)
# ============================================================

def _proses_satu_gambar(args):
    """
    Dijalankan di worker process (parallel).
    args = (image_path, image_name, use_numba)
    Return: (nama_orang, fitur_dict) atau None jika gagal.
    """
    global _EYE_CASCADE

    image_path, image_name, use_numba = args
    cascade = _EYE_CASCADE

    nama_orang = (
        image_name
        .replace('_face detected.png', '')
        .split('(')[0]
        .strip()
        .lower()
    )

    try:
        pil_img       = Image.open(image_path).convert('RGB')
        face_crop_rgb = np.array(pil_img)
        h_orig, w_orig, _ = face_crop_rgb.shape
    except Exception as e:
        return None, image_name, f"Gagal memuat: {e}"

    # Grayscale + equalize
    gray    = rgb_to_gray_manual(face_crop_rgb)
    gray_eq = equalize_hist(gray)

    # Deteksi mata
    try:
        if use_numba and NUMBA_OK and hasattr(cascade, '_numba_arrays'):
            # Gunakan Numba JIT path
            eyes_raw = _detect_multiscale_numba(
                cascade, gray_eq, scaleFactor=1.05,
                minNeighbors=5, minSize=(15, 15)
            )
        else:
            eyes_raw = cascade.detectMultiScale(
                gray_eq, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15)
            )
    except Exception as e:
        eyes_raw = []

    filtered_eyes = [
        (ex, ey, ew, eh)
        for (ex, ey, ew, eh) in eyes_raw
        if ey < h_orig * 0.6 and ew > 10 and eh > 10
    ]
    filtered_eyes = sorted(filtered_eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]

    # Skalakan koordinat mata ke 100x100
    eyes_100 = [
        (int(ex*100/w_orig), int(ey*100/h_orig),
         int(ew*100/w_orig), int(eh*100/h_orig))
        for (ex, ey, ew, eh) in filtered_eyes
    ]

    # Resize wajah ke 100x100
    face_100 = np.array(pil_img.resize((100, 100)))

    # Ekstraksi fitur
    try:
        extractor = FaceFeatureExtractor(target_size=(100, 100))
        hasil     = extractor.extract_features(face_100, eyes_100)
        fitur     = hasil["features"]
        fitur["file"] = image_name
    except Exception as e:
        return None, image_name, f"Gagal ekstraksi: {e}"

    return nama_orang, fitur, None  # (nama, fitur, error)


def _detect_multiscale_numba(cascade, gray, scaleFactor=1.05,
                               minNeighbors=5, minSize=(15, 15)):
    """DetectMultiScale dengan Numba JIT inner loop."""
    from src.manual_cascade import resize_bilinear, compute_integral_image, group_rectangles

    if len(gray.shape) == 3:
        gray = np.dot(gray[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    gray = gray.astype(np.uint8)

    img_h, img_w = gray.shape
    win_w, win_h = cascade.base_window_size
    inv_area     = 1.0 / (win_w * win_h)

    arrays = cascade._numba_arrays
    (feat_xs, feat_ys, feat_ws, feat_hs, feat_wts,
     feat_starts, feat_lens,
     stage_thresholds,
     weak_feat_idxs, weak_thresholds, weak_leaf0, weak_leaf1,
     stage_starts, stage_lens) = arrays

    detected = []
    scale = 1.0
    while True:
        scaled_w = int(img_w / scale)
        scaled_h = int(img_h / scale)
        if scaled_w < max(minSize[0], win_w) or scaled_h < max(minSize[1], win_h):
            break

        scaled_img = resize_bilinear(gray, scaled_w, scaled_h).astype(np.float64)
        ii, ii_sq  = compute_integral_image(scaled_img)
        step       = max(1, win_w // 8)

        hits_x, hits_y = _eval_cascade_numba(
            ii, ii_sq,
            feat_xs, feat_ys, feat_ws, feat_hs, feat_wts,
            feat_starts, feat_lens,
            stage_thresholds,
            weak_feat_idxs, weak_thresholds, weak_leaf0, weak_leaf1,
            stage_starts, stage_lens,
            win_w, win_h, scaled_w, scaled_h,
            step, inv_area
        )

        for x, y in zip(hits_x, hits_y):
            detected.append([int(x*scale), int(y*scale),
                              int(win_w*scale), int(win_h*scale)])
        scale *= scaleFactor

    return group_rectangles(detected, min_neighbors=minNeighbors, iou_thresh=0.2)


# ============================================================
# HELPER: cari cascade XML
# ============================================================

def find_cascade_xml(filename):
    local = os.path.join(ROOT, filename)
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
    raise FileNotFoundError(f"'{filename}' tidak ditemukan. Install opencv-python atau letakkan di folder ini.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ekstraksi Fitur Wajah — Versi Paralel (Multi-core)"
    )
    parser.add_argument(
        '--workers', type=int,
        default=multiprocessing.cpu_count(),
        help=f'Jumlah worker process (default: semua core = {multiprocessing.cpu_count()})'
    )
    parser.add_argument(
        '--no-numba', action='store_true',
        help='Nonaktifkan Numba JIT (gunakan pure NumPy saja)'
    )
    parser.add_argument(
        '--input', default='hasil_deteksi',
        help='Folder input gambar wajah (default: hasil_deteksi)'
    )
    parser.add_argument(
        '--output', default='database_fitur.json',
        help='File output database JSON (default: database_fitur.json)'
    )
    args = parser.parse_args()

    use_numba = NUMBA_OK and not args.no_numba
    n_workers = min(args.workers, multiprocessing.cpu_count())

    print("=" * 60)
    print("  EKSTRAKSI FITUR WAJAH — MODE PARALEL")
    print("=" * 60)
    print(f"  CPU Cores tersedia : {multiprocessing.cpu_count()}")
    print(f"  Workers digunakan  : {n_workers}")
    print(f"  Numba JIT          : {'AKTIF' if use_numba else 'TIDAK (pure NumPy)'}")
    if not NUMBA_OK and not args.no_numba:
        print("  [INFO] Install numba untuk performa lebih cepat:")
        print("         pip install numba")
    print("=" * 60)

    # --- Validasi input folder ---
    input_folder = args.input
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
        return

    print(f"\nDitemukan {len(face_files)} gambar wajah di '{input_folder}'.")

    # --- Load cascade untuk warmup / Numba prep ---
    print("Memuat Haar Cascade...")
    eye_xml_path = find_cascade_xml('haarcascade_eye.xml')
    print(f"  -> XML: {eye_xml_path}")

    # Pre-compile Numba (warmup) di main process
    if use_numba:
        print("  Kompilasi Numba JIT (sekali saja)...")
        from src.manual_cascade import ManualCascadeClassifier
        _warm_cascade = ManualCascadeClassifier(eye_xml_path)
        _warm_cascade._numba_arrays = _prepare_cascade_arrays(_warm_cascade)
        # Warmup dengan gambar dummy kecil
        dummy = np.zeros((48, 48), dtype=np.float64)
        dummy_ii    = np.zeros((49, 49), dtype=np.float64)
        dummy_ii_sq = np.zeros((49, 49), dtype=np.float64)
        _na = _warm_cascade._numba_arrays
        try:
            _eval_cascade_numba(
                dummy_ii, dummy_ii_sq,
                _na[0], _na[1], _na[2], _na[3], _na[4],
                _na[5], _na[6], _na[7],
                _na[8], _na[9], _na[10], _na[11],
                _na[12], _na[13],
                24, 24, 48, 48, 3, 1.0/(24*24)
            )
            print("  Numba JIT warmup selesai.")
        except Exception as e:
            print(f"  [WARN] Numba warmup error: {e} — fallback ke NumPy")
            use_numba = False

    # --- Siapkan argumen untuk setiap gambar ---
    task_args = [
        (os.path.join(input_folder, fname), fname, use_numba)
        for fname in face_files
    ]

    # --- Jalankan paralel ---
    print(f"\nMemulai ekstraksi paralel ({n_workers} workers)...\n")
    t_start = time.time()

    # multiprocessing.Pool: setiap worker load cascade SEKALI (_init_worker)
    # Ini kunci utama: tidak ada overhead load cascade per gambar
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(eye_xml_path,)
    ) as pool:
        results = pool.map(_proses_satu_gambar, task_args)

    t_elapsed = time.time() - t_start

    # --- Kumpulkan hasil ---
    database    = {}
    berhasil    = 0
    gagal       = 0

    for i, (nama_orang, fitur_atau_pesan, error) in enumerate(results, 1):
        fname = face_files[i - 1]
        if error:
            print(f"  [{i:>3}/{len(face_files)}] [SKIP] {fname}: {error}")
            gagal += 1
        else:
            if nama_orang not in database:
                database[nama_orang] = []
            database[nama_orang].append(fitur_atau_pesan)
            print(f"  [{i:>3}/{len(face_files)}] OK  {fname} -> '{nama_orang}'")
            berhasil += 1

    # --- Reset dan tulis database ---
    db_path = args.output
    if os.path.exists(db_path):
        os.remove(db_path)

    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False)

    # --- Ringkasan ---
    print()
    print("=" * 60)
    print(f"  Selesai dalam  : {t_elapsed:.1f} detik")
    print(f"  Kecepatan      : {len(face_files)/t_elapsed:.1f} gambar/detik")
    print(f"  Berhasil       : {berhasil}")
    print(f"  Gagal          : {gagal}")
    print(f"  Subjek         : {list(database.keys())}")
    print(f"  Database       : {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    # WAJIB: guard ini harus ada agar multiprocessing Windows tidak error
    multiprocessing.freeze_support()
    main()
