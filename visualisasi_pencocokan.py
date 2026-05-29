"""
visualisasi_pencocokan.py (Versi Rata-Rata Rumpun Fitur)
-------------------------------------------------------
File khusus Debugging yang menghitung rata-rata dari seluruh foto kandidat di database,
serta memplot semua titik fitur historis (maksimal 4 atau lebih) + 1 titik rata-ratanya.
"""

import os
import sys
import json
import glob
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from PIL import Image

from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.feature_extractor import FaceFeatureExtractor
from src.pencocokan import FaceMatcher

def find_cascade_xml(filename):
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(local): return local
    try:
        import site
        all_site = site.getsitepackages() + [site.getusersitepackages()]
        for sp in all_site:
            p = os.path.join(sp, 'cv2', 'data', filename)
            if os.path.exists(p): return p
    except Exception: pass
    return filename

# ============================================================
# FUNGSI BARU: Menggambar 4 Titik Histori + 1 Titik Rata-rata
# ============================================================
def gambar_struktur_wajah_multititik(ax, img_100, list_fitur, fitur_rata_rata):
    ax.imshow(img_100)
    ax.axis('off')
    
    pe = [path_effects.withStroke(linewidth=2, foreground='black')]
    
    # 1. Plot ke-4 (atau semua) titik fitur historis dari database dengan warna redup/transparan
    for f in list_fitur:
        eye_coords = f.get("eye_coords", [])
        nose_coords = f.get("nose_coords", [])
        mouth_coords = f.get("mouth_coords", [])
        
        # Titik-titik kecil transparan (Histori gambar 1-4)
        for ec in eye_coords:
            ax.plot(ec[0], ec[1], 'ro', markersize=4, alpha=0.4)
        for nc in nose_coords:
            ax.plot(nc[0], nc[1], 'bo', markersize=4, alpha=0.4)
        for mc in mouth_coords:
            ax.plot(mc[0], mc[1], 'go', markersize=4, alpha=0.4)

    # 2. Plot 1 Titik RATA-RATA yang tebal dan jelas di atasnya
    avg_eyes = fitur_rata_rata.get("eye_coords", [])
    avg_nose = fitur_rata_rata.get("nose_coords", [])
    avg_mouth = fitur_rata_rata.get("mouth_coords", [])
    avg_dist = fitur_rata_rata.get("eye_distance", 0)
    
    if len(avg_eyes) == 2:
        c1, c2 = avg_eyes
        # Mata Rata-rata (Bintang Kuning/Merah Terang bergaris)
        ax.plot(c1[0], c1[1], 'r*', markersize=10, markeredgecolor='white', label='Rata-rata')
        ax.plot(c2[0], c2[1], 'r*', markersize=10, markeredgecolor='white')
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'r--', linewidth=2)
        ax.text((c1[0]+c2[0])/2, ((c1[1]+c2[1])/2)-5, f"Avg D:{avg_dist:.1f}", 
                color='yellow', ha='center', fontsize=9, path_effects=pe, fontweight='bold')
        
    if len(avg_nose) == 1:
        ax.plot(avg_nose[0][0], avg_nose[0][1], 'b*', markersize=10, markeredgecolor='white')
        
    if len(avg_mouth) == 1:
        ax.plot(avg_mouth[0][0], avg_mouth[0][1], 'g*', markersize=10, markeredgecolor='white')

def main():
    parser = argparse.ArgumentParser(description="Visual Debugger Rata-Rata Fitur Wajah")
    parser.add_argument("gambar", help="Path ke file gambar input")
    parser.add_argument("--threshold", "-t", type=float, default=None)
    parser.add_argument("--metode", "-m", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    parser.add_argument("--database", "-d", default="database_fitur.json")
    args = parser.parse_args()

    if not os.path.exists(args.gambar):
        print(f"[ERROR] Gambar input '{args.gambar}' tidak ditemukan!")
        sys.exit(1)

    # Load Pipeline
    face_cascade = ManualCascadeClassifier(find_cascade_xml('haarcascade_frontalface_default.xml'))
    eye_cascade  = ManualCascadeClassifier(find_cascade_xml('haarcascade_eye.xml'))
    extractor = FaceFeatureExtractor(target_size=(100, 100))
    matcher = FaceMatcher(db_path=args.database)

    # Proses Gambar Input Saat Ini
    pil_img = Image.open(args.gambar).convert('RGB')
    img_orig = np.array(pil_img)
    h_frame, w_frame, _ = img_orig.shape
    
    gray = rgb_to_gray_manual(img_orig)
    gray_eq = equalize_hist(gray)
    faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

    if len(faces) == 0:
        face_crop = img_orig
        x, y, w, h_ext = 0, 0, w_frame, h_frame
    else:
        x, y, w, h = faces[0]
        h_ext = min(int(h * 1.25), h_frame - y)
        face_crop = img_orig[y:y + h_ext, x:x + w]

    h_crop, w_crop, _ = face_crop.shape
    gray_crop_eq = equalize_hist(rgb_to_gray_manual(face_crop))
    eyes_raw = eye_cascade.detectMultiScale(gray_crop_eq, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15))
    filtered_eyes = sorted([e for e in eyes_raw if e[1] < h_crop * 0.6], key=lambda e: e[2]*e[3], reverse=True)[:2]
    eyes_100 = [(int(ex*100/w_crop), int(ey*100/h_crop), int(ew*100/w_crop), int(eh*100/h_crop)) for (ex, ey, ew, eh) in filtered_eyes]

    face_input_100 = np.array(Image.fromarray(face_crop).resize((100, 100)))
    fitur_input = extractor.extract_features(face_input_100, eyes_100)["features"]

    # Jalankan Matcher untuk menentukan siapa kandidat terdekat
    hasil_match = matcher.match(fitur_input, threshold=args.threshold, metode=args.metode)
    kandidat_nama = hasil_match["nama_kandidat"]
    
    # ============================================================
    # PROSES UTAMA: HITUNG RATA-RATA KOMPONEN DATABASE (Saran Teman)
    # ============================================================
    with open(args.database, 'r') as f:
        db_raw = json.load(f)
    
    all_snapshots = db_raw[kandidat_nama] # Ambil seluruh histori gambar (misal ada 4)
    print(f"\n[INFO] Menghitung rata-rata dari {len(all_snapshots)} sampel foto milik '{kandidat_nama}' di database.")

    # Dictionary untuk menampung hasil rata-rata
    fitur_rata_rata = {}
    kunci_numerik = ["eye_distance", "skin_ratio", "edge_density", "contour_density", "symmetry_error"]
    
    for k in kunci_numerik:
        fitur_rata_rata[k] = np.mean([float(snap.get(k, 0)) for snap in all_snapshots])

    # Hitung rata-rata koordinat posisi secara manual (Mata, Hidung, Mulut)
    def ambil_rata_rata_koordinat(list_of_coords_lists):
        # Menyaring data koordinat yang valid (tidak kosong)
        valid_coords = [c for c in list_of_coords_lists if len(c) > 0]
        if not valid_coords: return []
        return np.mean(valid_coords, axis=0).astype(int).tolist()

    fitur_rata_rata["eye_coords"] = ambil_rata_rata_koordinat([snap.get("eye_coords", []) for snap in all_snapshots])
    fitur_rata_rata["nose_coords"] = ambil_rata_rata_koordinat([snap.get("nose_coords", []) for snap in all_snapshots])
    fitur_rata_rata["mouth_coords"] = ambil_rata_rata_koordinat([snap.get("mouth_coords", []) for snap in all_snapshots])

    # Ambil salah satu gambar database sebagai background visualisasi
    search_pattern = os.path.join("hasil_deteksi", f"{kandidat_nama}*_face detected.png")
    matched_db_images = glob.glob(search_pattern)
    img_db_100 = cv2.resize(cv2.cvtColor(cv2.imread(matched_db_images[0]), cv2.COLOR_BGR2RGB), (100, 100)) if matched_db_images else (np.zeros((100,100,3), dtype=np.uint8)+100)

    # ============================================================
    # RENDER MATPLOTLIB VISUALISASI
    # ============================================================
    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title(f"DEBUGGER RATA-RATA TEMPLATE ({args.metode.upper()})")

    # Panel 1: Gambar Asli yang sedang diinputkan
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("1. Gambar Input Asli", fontsize=12, fontweight='bold')
    ax1.imshow(img_orig)
    ax1.axis('off')
    if len(faces) > 0:
        ax1.add_patch(patches.Rectangle((x, y), w, h_ext, linewidth=2, edgecolor='g', facecolor='none'))

    # Panel 2: Titik Fitur tunggal dari Gambar Input saat ini
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title("2. Fitur Gambar Input Saat Ini", fontsize=12, fontweight='bold')
    # Pakai fungsi dummy bungkus ke list agar kompatibel dengan pemetaan gambar tunggal
    gambar_struktur_wajah_multititik(ax2, face_input_100, [fitur_input], fitur_input)

    # Panel 3: CLUSTER TITIK (4 Histori Titik Kecil + 1 Bintang Rata-Rata) -> Ini permintaan temanmu!
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title(f"3. Kluster Fitur DB: '{kandidat_nama.upper()}' ({len(all_snapshots)} Foto)", fontsize=12, fontweight='bold')
    gambar_struktur_wajah_multititik(ax3, img_db_100, all_snapshots, fitur_rata_rata)

    # Panel 4: Bar Chart Komparasi Nilai Rata-rata vs Input
    ax_chart = plt.subplot(2, 3, (4, 5))
    ax_chart.set_title("Perbandingan Angka Karakteristik (Input vs Rata-rata DB)", fontsize=12, fontweight='bold')
    
    labels_piagam = ["Jarak Mata", "Rasio Kulit", "Kerapatan Tepi", "Kontur Kulit", "Error Simetri"]
    nilai_input = [float(fitur_input.get(k, 0)) for k in kunci_numerik]
    nilai_db_avg = [float(fitur_rata_rata.get(k, 0)) for k in kunci_numerik]
    
    y_pos = np.arange(len(labels_piagam))
    width = 0.35
    ax_chart.barh(y_pos - width/2, nilai_input, width, label='Gambar Input Baru', color='#3498db')
    ax_chart.barh(y_pos + width/2, nilai_db_avg, width, label='Rata-Rata Database', color='#f1c40f')
    ax_chart.set_yticks(y_pos)
    ax_chart.set_yticklabels(labels_piagam)
    ax_chart.invert_yaxis()
    ax_chart.legend()

    # Panel 5: Status Verifikasi Kelulusan
    ax_status = plt.subplot(2, 3, 6)
    ax_status.axis('off')
    status_text = "DIKENALI" if hasil_match["dikenali"] else "TIDAK DIKENALI"
    box_style = dict(boxstyle='round,pad=1', facecolor='#f5f5f5', edgecolor='gray')
    ax_status.text(0.5, 0.5, 
                   f"STATUS KEPUTUSAN:\n"
                   f"{'✅' if hasil_match['dikenali'] else '❌'} {status_text}\n\n"
                   f"Kandidat Terdekat:\n"
                   f"👉 {kandidat_nama.upper()}\n\n"
                   f"Jarak Skor Akhir:\n"
                   f"📊 {hasil_match['jarak']:.4f}",
                   fontsize=12, fontweight='bold', ha='center', va='center', bbox=box_style)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()