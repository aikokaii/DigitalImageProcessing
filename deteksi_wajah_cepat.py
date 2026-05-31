"""
deteksi_wajah_cepat.py
=======================
Versi PARALEL dari deteksi_wajah.py menggunakan semua CPU core.

Strategi percepatan:
  1. multiprocessing.Pool — N gambar diproses paralel (semua core)
  2. Cascade (face + eye) diload SEKALI per worker process, bukan per gambar
  3. Numba JIT pada inner loop cascade (jika numba terinstall)
  4. Resize ke 320x240 tetap dilakukan untuk konsistensi dengan versi lama

Install numba (opsional, tapi bikin 10-50x lebih cepat per gambar):
    pip install numba

Cara pakai:
    python deteksi_wajah_cepat.py
    python deteksi_wajah_cepat.py --workers 8
    python deteksi_wajah_cepat.py --input webcam --no-numba
"""

import os
import sys
import time
import argparse
import multiprocessing
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.face_utils import gaussian_blur_manual, sobel_edge_detection, skin_segmentation_manual
from src.image_utils import equalize_hist, rgb_to_gray_manual

# ── Numba (opsional) ──────────────────────────────────────────────────────
try:
    from numba import njit
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False

# ── Global per-worker (diload sekali saat worker init) ────────────────────
_FACE_CASCADE = None
_EYE_CASCADE  = None
_USE_NUMBA    = False
_FACE_ARRAYS  = None
_EYE_ARRAYS   = None


# ============================================================
# NUMBA JIT — inner loop cascade (dikompilasi jadi native code)
# ============================================================

if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _cascade_inner(ii, ii_sq,
                       feat_xs, feat_ys, feat_ws, feat_hs, feat_wts,
                       feat_starts, feat_lens,
                       stage_thresh,
                       wk_fidx, wk_thresh, wk_leaf0, wk_leaf1,
                       stage_starts, stage_lens,
                       win_w, win_h, scaled_w, scaled_h, step, inv_area):
        hits_x = []
        hits_y = []
        for y in range(0, scaled_h - win_h, step):
            for x in range(0, scaled_w - win_w, step):
                ws  = ii[y+win_h, x+win_w] - ii[y, x+win_w] - ii[y+win_h, x] + ii[y, x]
                wss = ii_sq[y+win_h, x+win_w] - ii_sq[y, x+win_w] - ii_sq[y+win_h, x] + ii_sq[y, x]
                mean = ws * inv_area
                var  = wss * inv_area - mean * mean
                std  = var**0.5 if var > 0.0 else 1.0

                ok = True
                for si in range(len(stage_thresh)):
                    ssum = 0.0
                    ws_i = stage_starts[si]
                    we_i = ws_i + stage_lens[si]
                    for wi in range(ws_i, we_i):
                        fi   = wk_fidx[wi]
                        fval = 0.0
                        fs   = feat_starts[fi]
                        fe   = fs + feat_lens[fi]
                        for ri in range(fs, fe):
                            rx = feat_xs[ri]; ry = feat_ys[ri]
                            rw = feat_ws[ri]; rh = feat_hs[ri]
                            rweight = feat_wts[ri]
                            if (x+rx+rw) <= scaled_w and (y+ry+rh) <= scaled_h:
                                rs = ii[y+ry+rh, x+rx+rw] - ii[y+ry, x+rx+rw] \
                                   - ii[y+ry+rh, x+rx] + ii[y+ry, x+rx]
                                fval += rs * rweight
                        fval *= inv_area
                        ssum += wk_leaf0[wi] if fval < wk_thresh[wi] * std else wk_leaf1[wi]
                    if ssum < stage_thresh[si]:
                        ok = False
                        break
                if ok:
                    hits_x.append(x)
                    hits_y.append(y)
        return hits_x, hits_y
else:
    _cascade_inner = None


def _build_arrays(cascade):
    """Konversi cascade ke flat numpy array untuk Numba."""
    feat_xs, feat_ys, feat_ws, feat_hs, feat_wts = [], [], [], [], []
    feat_starts, feat_lens = [], []
    pos = 0
    for rects in cascade.features:
        feat_starts.append(pos)
        feat_lens.append(len(rects))
        for rx, ry, rw, rh, rweight in rects:
            feat_xs.append(rx); feat_ys.append(ry)
            feat_ws.append(rw); feat_hs.append(rh)
            feat_wts.append(rweight)
        pos += len(rects)

    s_thresh, s_starts, s_lens = [], [], []
    wk_fidx, wk_thresh, wk_l0, wk_l1 = [], [], [], []
    wp = 0
    for stage in cascade.stages:
        s_thresh.append(stage['threshold'])
        s_starts.append(wp)
        s_lens.append(len(stage['weak_classifiers']))
        for wc in stage['weak_classifiers']:
            wk_fidx.append(wc['feature_idx'])
            wk_thresh.append(wc['threshold'])
            wk_l0.append(wc['leaf0'])
            wk_l1.append(wc['leaf1'])
        wp += len(stage['weak_classifiers'])

    return (
        np.array(feat_xs,    np.int32),
        np.array(feat_ys,    np.int32),
        np.array(feat_ws,    np.int32),
        np.array(feat_hs,    np.int32),
        np.array(feat_wts,   np.float64),
        np.array(feat_starts,np.int32),
        np.array(feat_lens,  np.int32),
        np.array(s_thresh,   np.float64),
        np.array(wk_fidx,    np.int32),
        np.array(wk_thresh,  np.float64),
        np.array(wk_l0,      np.float64),
        np.array(wk_l1,      np.float64),
        np.array(s_starts,   np.int32),
        np.array(s_lens,     np.int32),
    )


def _detect_fast(cascade, arrays, gray,
                 scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)):
    """
    DetectMultiScale menggunakan Numba JIT (jauh lebih cepat dari pure Python).
    """
    from src.manual_cascade import resize_bilinear, compute_integral_image, group_rectangles

    if len(gray.shape) == 3:
        gray = np.dot(gray[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    gray = gray.astype(np.uint8)

    img_h, img_w = gray.shape
    win_w, win_h = cascade.base_window_size
    inv_area = 1.0 / (win_w * win_h)

    (feat_xs, feat_ys, feat_ws, feat_hs, feat_wts,
     feat_starts, feat_lens,
     s_thresh,
     wk_fidx, wk_thresh, wk_l0, wk_l1,
     s_starts, s_lens) = arrays

    detected = []
    scale = 1.0
    while True:
        sw = int(img_w / scale)
        sh = int(img_h / scale)
        if sw < max(minSize[0], win_w) or sh < max(minSize[1], win_h):
            break

        scaled = resize_bilinear(gray, sw, sh).astype(np.float64)
        ii, ii_sq = compute_integral_image(scaled)
        step = max(1, win_w // 8)

        hx, hy = _cascade_inner(
            ii, ii_sq,
            feat_xs, feat_ys, feat_ws, feat_hs, feat_wts,
            feat_starts, feat_lens,
            s_thresh,
            wk_fidx, wk_thresh, wk_l0, wk_l1,
            s_starts, s_lens,
            win_w, win_h, sw, sh, step, inv_area
        )
        for x, y in zip(hx, hy):
            detected.append([int(x*scale), int(y*scale),
                              int(win_w*scale), int(win_h*scale)])
        scale *= scaleFactor

    return group_rectangles(detected, min_neighbors=minNeighbors, iou_thresh=0.2)


# ============================================================
# WORKER INIT — dijalankan sekali saat pool dibuat
# ============================================================

def _init_worker(face_xml, eye_xml, use_numba):
    global _FACE_CASCADE, _EYE_CASCADE, _USE_NUMBA, _FACE_ARRAYS, _EYE_ARRAYS
    from src.manual_cascade import ManualCascadeClassifier

    _FACE_CASCADE = ManualCascadeClassifier(face_xml)
    _EYE_CASCADE  = ManualCascadeClassifier(eye_xml)
    _USE_NUMBA    = use_numba and NUMBA_OK

    if _USE_NUMBA:
        _FACE_ARRAYS = _build_arrays(_FACE_CASCADE)
        _EYE_ARRAYS  = _build_arrays(_EYE_CASCADE)


# ============================================================
# FUNGSI PROSES 1 GAMBAR (dijalankan di worker)
# ============================================================

def _proses_satu(args):
    """
    Proses 1 gambar: load → preprocess → deteksi wajah → deteksi mata
    → skin/edge → simpan. Return: dict statistik.
    """
    global _FACE_CASCADE, _EYE_CASCADE, _USE_NUMBA, _FACE_ARRAYS, _EYE_ARRAYS

    (image_name, image_path,
     output_detection, output_skin, output_edge,
     datatest_dir, datatest_skin, datatest_edge) = args

    stat = {
        'name'         : image_name,
        'face_detected': False,
        'eyes_found'   : 0,
        'eyes_expected': 0,
        'error'        : None,
    }

    # --- Load gambar ---
    try:
        pil_img = Image.open(image_path).convert('RGB')
        pil_img = pil_img.resize((320, 240))
        frame   = np.array(pil_img)
    except Exception as e:
        stat['error'] = str(e)
        return stat

    # --- Preprocessing ---
    gray = rgb_to_gray_manual(frame)
    gray = equalize_hist(gray)

    # --- Deteksi wajah ---
    if _USE_NUMBA:
        raw_faces = _detect_fast(_FACE_CASCADE, _FACE_ARRAYS, gray,
                                  scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
    else:
        raw_faces = _FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))

    faces = [(x,y,w,h) for (x,y,w,h) in raw_faces if w >= 60 and h >= 60]

    if not faces:
        stat['face_detected'] = False
        return stat  # buang — tidak disimpan

    stat['face_detected'] = True

    # --- Tentukan folder output ---
    if 'datatest' in image_name.lower():
        out_det  = datatest_dir
        out_skin = datatest_skin
        out_edge = datatest_edge
    else:
        out_det  = output_detection
        out_skin = output_skin
        out_edge = output_edge

    for d in [out_det, out_skin, out_edge]:
        os.makedirs(d, exist_ok=True)

    filename = os.path.splitext(image_name)[0]
    skin_mask_save = None
    edges_save     = None
    final_crop     = frame

    # --- Loop per wajah ---
    for (x, y, w, h) in faces:
        h_new  = min(int(h * 1.25), frame.shape[0] - y)
        roi_gray  = gray[y:y+h_new, x:x+w]
        roi_color = frame[y:y+h_new, x:x+w]
        final_crop = roi_color

        # Blur + skin + edge
        blurred   = gaussian_blur_manual(roi_gray)

        pil_crop  = Image.fromarray(roi_color)
        pil_hsv   = np.array(pil_crop.convert('HSV'))
        hsv       = np.zeros_like(pil_hsv)
        hsv[:,:,0] = (pil_hsv[:,:,0].astype(np.float32) * 179/255).astype(np.uint8)
        hsv[:,:,1] = pil_hsv[:,:,1]
        hsv[:,:,2] = pil_hsv[:,:,2]

        skin_mask_save = skin_segmentation_manual(hsv)
        edges_save     = sobel_edge_detection(blurred)

        # --- Deteksi mata ---
        if _USE_NUMBA:
            eyes_raw = _detect_fast(_EYE_CASCADE, _EYE_ARRAYS, roi_gray,
                                     scaleFactor=1.05, minNeighbors=9, minSize=(20,20))
        else:
            eyes_raw = _EYE_CASCADE.detectMultiScale(
                roi_gray, scaleFactor=1.05, minNeighbors=9, minSize=(20,20))

        filtered_eyes = sorted(
            [(ex,ey,ew,eh) for (ex,ey,ew,eh) in eyes_raw
             if ey < h_new*0.5 and ew > 15 and eh > 15],
            key=lambda e: e[2]*e[3], reverse=True
        )[:2]

        stat['eyes_expected'] += 2
        stat['eyes_found']    += len(filtered_eyes)

    # --- Simpan hasil ---
    if skin_mask_save is not None:
        Image.fromarray(skin_mask_save, mode='L').save(
            os.path.join(out_skin, filename + '_skin.png'))
        Image.fromarray(edges_save, mode='L').save(
            os.path.join(out_edge, filename + '_edge.png'))

    Image.fromarray(final_crop, mode='RGB').save(
        os.path.join(out_det, filename + '_face detected.png'))

    return stat


# ============================================================
# HELPER: cari cascade XML
# ============================================================

def _find_xml(filename):
    local = os.path.join(ROOT, filename)
    if os.path.exists(local):
        return local
    try:
        import site
        paths = site.getsitepackages()
        try:
            paths += [site.getusersitepackages()]
        except Exception:
            pass
        for sp in paths:
            p = os.path.join(sp, 'cv2', 'data', filename)
            if os.path.exists(p):
                return p
    except Exception:
        pass
    raise FileNotFoundError(
        f"'{filename}' tidak ditemukan.\n"
        "Pastikan opencv-python terinstall: pip install opencv-python\n"
        "ATAU letakkan file XML di folder yang sama dengan script ini."
    )


# ============================================================
# WARMUP NUMBA
# ============================================================

def _warmup_numba(face_xml, eye_xml):
    """Kompilasi JIT sekali di main process agar worker langsung siap."""
    from src.manual_cascade import ManualCascadeClassifier
    from src.manual_cascade import compute_integral_image

    print("  Kompilasi Numba JIT (sekali, ~10-30 detik)...")
    fc = ManualCascadeClassifier(face_xml)
    fa = _build_arrays(fc)
    ec = ManualCascadeClassifier(eye_xml)
    ea = _build_arrays(ec)

    dummy = np.zeros((48, 48), np.float64)
    ii, iisq = compute_integral_image(dummy)

    for arr in [fa, ea]:
        (fxs, fys, fws, fhs, fwts, fst, fle,
         sth, wfi, wth, wl0, wl1, sst, sle) = arr
        try:
            _cascade_inner(
                ii, iisq,
                fxs, fys, fws, fhs, fwts, fst, fle,
                sth, wfi, wth, wl0, wl1, sst, sle,
                24, 24, 48, 48, 3, 1.0/(24*24)
            )
        except Exception:
            return False  # numba error, fallback

    print("  Numba JIT warmup selesai.")
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deteksi Wajah Haar Cascade — Versi Paralel Multi-core"
    )
    parser.add_argument('--input',    default='webcam',
                        help='Folder input gambar (default: webcam)')
    parser.add_argument('--workers',  type=int,
                        default=multiprocessing.cpu_count(),
                        help=f'Jumlah worker (default: {multiprocessing.cpu_count()} core)')
    parser.add_argument('--no-numba', action='store_true',
                        help='Nonaktifkan Numba JIT')
    args = parser.parse_args()

    use_numba = NUMBA_OK and not args.no_numba
    n_workers = min(args.workers, multiprocessing.cpu_count())

    # --- Header ---
    print("=" * 62)
    print("  DETEKSI WAJAH HAAR CASCADE — MODE PARALEL")
    print("=" * 62)
    print(f"  CPU cores tersedia : {multiprocessing.cpu_count()}")
    print(f"  Workers digunakan  : {n_workers}")
    print(f"  Numba JIT          : {'AKTIF' if use_numba else 'TIDAK (pure NumPy)'}")
    if not NUMBA_OK and not args.no_numba:
        print("  [INFO] pip install numba  untuk akselerasi lebih lanjut")
    print("=" * 62)

    # --- Cari XML ---
    print("\nMencari file Haar Cascade XML...")
    face_xml = _find_xml('haarcascade_frontalface_default.xml')
    eye_xml  = _find_xml('haarcascade_eye.xml')
    print(f"  Face: {face_xml}")
    print(f"  Eye : {eye_xml}")

    # --- Warmup Numba (di main process, sekali) ---
    if use_numba:
        ok = _warmup_numba(face_xml, eye_xml)
        if not ok:
            print("  [WARN] Numba warmup gagal, fallback ke pure NumPy")
            use_numba = False

    # --- Validasi folder input ---
    input_folder = args.input
    if not os.path.exists(input_folder):
        print(f"\n[ERROR] Folder '{input_folder}' tidak ditemukan!")
        sys.exit(1)

    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm', '.bmp'))
    ])

    if not image_files:
        print(f"[ERROR] Tidak ada gambar di '{input_folder}'.")
        sys.exit(1)

    print(f"\nDitemukan {len(image_files)} gambar di '{input_folder}'.")

    # --- Folder output ---
    OUT_DET       = 'hasil_deteksi'
    OUT_SKIN      = 'hasil_skin'
    OUT_EDGE      = 'hasil_edge'
    DATATEST_DIR  = 'datatest'
    DATATEST_SKIN = 'datatest_skin'
    DATATEST_EDGE = 'datatest_edge'

    for d in [OUT_DET, OUT_SKIN, OUT_EDGE]:
        os.makedirs(d, exist_ok=True)

    # --- Susun task args ---
    task_args = [
        (fname,
         os.path.join(input_folder, fname),
         OUT_DET, OUT_SKIN, OUT_EDGE,
         DATATEST_DIR, DATATEST_SKIN, DATATEST_EDGE)
        for fname in image_files
    ]

    # --- Jalankan paralel ---
    print(f"\nMemulai deteksi paralel ({n_workers} workers)...\n")
    t0 = time.time()

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(face_xml, eye_xml, use_numba)
    ) as pool:
        results = pool.map(_proses_satu, task_args)

    elapsed = time.time() - t0

    # --- Tampilkan hasil ---
    total          = len(results)
    face_detected  = sum(1 for r in results if r['face_detected'])
    face_failed    = sum(1 for r in results if not r['face_detected'] and not r['error'])
    errors         = sum(1 for r in results if r['error'])
    eyes_found     = sum(r['eyes_found']    for r in results)
    eyes_expected  = sum(r['eyes_expected'] for r in results)

    print()
    for r in results:
        if r['error']:
            print(f"  [ERROR] {r['name']}: {r['error']}")
        elif r['face_detected']:
            print(f"  [OK]    {r['name']} — wajah terdeteksi, "
                  f"mata: {r['eyes_found']}/{r['eyes_expected']}")
        else:
            print(f"  [SKIP]  {r['name']} — wajah tidak terdeteksi, gambar dibuang")

    # --- Ringkasan ---
    print()
    print("=" * 62)
    print("  HASIL EVALUASI")
    print("=" * 62)
    print(f"  Total gambar               : {total}")
    print(f"  Wajah berhasil terdeteksi  : {face_detected}")
    print(f"  Wajah gagal terdeteksi     : {face_failed}")
    print(f"  Error (gagal baca gambar)  : {errors}")
    if total > 0:
        print(f"  Akurasi deteksi wajah      : {face_detected/total*100:.2f}%")
    if eyes_expected > 0:
        print(f"  Mata terdeteksi            : {eyes_found}/{eyes_expected} "
              f"({eyes_found/eyes_expected*100:.1f}%)")
    print(f"  Waktu total                : {elapsed:.1f} detik")
    print(f"  Kecepatan                  : {total/elapsed:.1f} gambar/detik")
    print("=" * 62)


if __name__ == '__main__':
    # WAJIB di Windows agar multiprocessing tidak spawn ulang tanpa henti
    multiprocessing.freeze_support()
    main()
