import os
import sys
import json
import glob
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from PIL import Image

from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.feature_extractor import FaceFeatureExtractor
from src.pencocokan import FaceMatcher

# ============================================================
# HELPER: Cari path XML Haar Cascade
# ============================================================
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
# MAIN FUNCTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Analisis Kluster Koordinat Fitur Wajah")
    parser.add_argument("gambar", help="Path ke file gambar input untuk diuji")
    parser.add_argument("--database", "-d", default="database_fitur.json")
    parser.add_argument("--metode", "-m", choices=["euclidean", "manhattan"], default="euclidean")
    args = parser.parse_args()

    if not os.path.exists(args.gambar):
        print(f"[ERROR] Gambar input '{args.gambar}' tidak ditemukan!")
        sys.exit(1)

    # 1. Inisialisasi Modul
    face_cascade = ManualCascadeClassifier(find_cascade_xml('haarcascade_frontalface_default.xml'))
    eye_cascade  = ManualCascadeClassifier(find_cascade_xml('haarcascade_eye.xml'))
    extractor = FaceFeatureExtractor(target_size=(100, 100))
    matcher = FaceMatcher(db_path=args.database)

    # 2. Proses Ekstraksi Gambar Input
    pil_img = Image.open(args.gambar).convert('RGB')
    img_orig = np.array(pil_img)
    h_frame, w_frame, _ = img_orig.shape

    gray_eq = equalize_hist(rgb_to_gray_manual(img_orig))
    faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

    if len(faces) == 0:
        face_crop = img_orig
        w_crop, h_crop = w_frame, h_frame
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

    # 3. Ambil Nama Kandidat Terdekat dari Database
    hasil_match = matcher.match(fitur_input, metode=args.metode)
    kandidat_nama = hasil_match["nama_kandidat"]

    # 4. Ambil SELURUH Histori Foto Kandidat (Misal 5 Foto) & Hitung Rata-rata koordinat
    with open(args.database, 'r') as f:
        db_raw = json.load(f)
    
    all_snapshots = db_raw[kandidat_nama]
    n_samples = len(all_snapshots)
    print(f"[INFO] Ditemukan {n_samples} sampel data untuk '{kandidat_nama}' di database. Memproses nilai rata-rata...")

    # Kumpulan koordinat terpisah dari database untuk plotting grafis persebaran
    db_mata_kiri_x, db_mata_kiri_y = [], []
    db_mata_kanan_x, db_mata_kanan_y = [], []
    db_hidung_x, db_hidung_y = [], []
    db_mulut_x, db_mulut_y = [], []

    for snap in all_snapshots:
        eyes = snap.get("eye_coords", [])
        nose = snap.get("nose_coords", [])
        mouth = snap.get("mouth_coords", [])

        if len(eyes) == 2:
            # Urutkan berdasarkan koordinat X untuk memisahkan mata kiri dan mata kanan
            eyes_sorted = sorted(eyes, key=lambda c: c[0])
            db_mata_kiri_x.append(eyes_sorted[0][0])
            db_mata_kiri_y.append(eyes_sorted[0][1])
            db_mata_kanan_x.append(eyes_sorted[1][0])
            db_mata_kanan_y.append(eyes_sorted[1][1])
        if len(nose) == 1:
            db_hidung_x.append(nose[0][0])
            db_hidung_y.append(nose[0][1])
        if len(mouth) == 1:
            db_mulut_x.append(mouth[0][0])
            db_mulut_y.append(mouth[0][1])

    # Hitung nilai Mean (Rata-rata) koordinat database
    avg_fitur = {
        "mata_kiri": [np.mean(db_mata_kiri_x), np.mean(db_mata_kiri_y)] if db_mata_kiri_x else [0,0],
        "mata_kanan": [np.mean(db_mata_kanan_x), np.mean(db_mata_kanan_y)] if db_mata_kanan_x else [0,0],
        "hidung": [np.mean(db_hidung_x), np.mean(db_hidung_y)] if db_hidung_x else [0,0],
        "mulut": [np.mean(db_mulut_x), np.mean(db_mulut_y)] if db_mulut_x else [0,0]
    }

    # Ambil salah satu gambar database yang valid untuk background gambar pembanding
    search_pattern = os.path.join("hasil_deteksi", f"{kandidat_nama}*_face detected.png")
    matched_db_images = glob.glob(search_pattern)
    if matched_db_images:
        img_db_100 = cv2.resize(cv2.cvtColor(cv2.imread(matched_db_images[0]), cv2.COLOR_BGR2RGB), (100, 100))
    else:
        img_db_100 = np.zeros((100, 100, 3), dtype=np.uint8) + 120

    # ============================================================
    # 5. MATPLOTLIB: RENDER VISUALISASI SEBARAN & KLUSTER KOORDINAT
    # ============================================================
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title(f"Analisis Kluster Sebaran Koordinat Wajah - {kandidat_nama.upper()}")
    pe = [path_effects.withStroke(linewidth=3, foreground='black')]

    # --- PANEL 1: Gambar Wajah Input ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(face_input_100)
    ax1.set_title("1. Wajah Input (Deteksi Saat Ini)", fontsize=12, fontweight='bold', pad=10)
    ax1.axis('off')
    # Plot koordinat wajah input saat ini
    inp_eyes = sorted(fitur_input.get("eye_coords", []), key=lambda c: c[0])
    inp_nose = fitur_input.get("nose_coords", [])
    inp_mouth = fitur_input.get("mouth_coords", [])
    
    for e_pt in inp_eyes: ax1.plot(e_pt[0], e_pt[1], 'ro', markersize=6, markeredgecolor='w')
    if inp_nose: ax1.plot(inp_nose[0][0], inp_nose[0][1], 'bo', markersize=6, markeredgecolor='w')
    if inp_mouth: ax1.plot(inp_mouth[0][0], inp_mouth[0][1], 'go', markersize=6, markeredgecolor='w')


    # --- PANEL 2: Gambar Model Database Terdekat (Template Overlay) ---
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(img_db_100)
    ax2.set_title(f"2. Representasi Wajah DB: '{kandidat_nama.upper()}' ({n_samples} Sampel)", fontsize=12, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # Plot Semua Histori Titik di Atas Gambar (Transparan/Kecil)
    ax2.scatter(db_mata_kiri_x + db_mata_kanan_x, db_mata_kiri_y + db_mata_kanan_y, c='red', s=15, alpha=0.4, label='Histori Mata')
    ax2.scatter(db_hidung_x, db_hidung_y, c='blue', s=15, alpha=0.4, label='Histori Hidung')
    ax2.scatter(db_mulut_x, db_mulut_y, c='green', s=15, alpha=0.4, label='Histori Mulut')
    
    # Plot Titik Rata-rata (Bintang Besar)
    ax2.plot(avg_fitur["mata_kiri"][0], avg_fitur["mata_kiri"][1], 'r*', markersize=12, markeredgecolor='w')
    ax2.plot(avg_fitur["mata_kanan"][0], avg_fitur["mata_kanan"][1], 'r*', markersize=12, markeredgecolor='w')
    ax2.plot(avg_fitur["hidung"][0], avg_fitur["hidung"][1], 'b*', markersize=12, markeredgecolor='w')
    ax2.plot(avg_fitur["mulut"][0], avg_fitur["mulut"][1], 'g*', markersize=12, markeredgecolor='w')


    # --- PANEL 3 & 4 GABUNG (BAWAH): Grafik Koordinat Sebaran Titik (Scatter Plot 2D) ---
    ax_graph = plt.subplot(2, 2, (3, 4))
    ax_graph.set_title("3. Grafik Koordinat Persebaran Titik Komponen Wajah (Skala Ruang 100x100)", fontsize=13, fontweight='bold', pad=15)
    
    # Grid & Batas Kanvas Gambar (0 sampai 100)
    ax_graph.set_xlim(0, 100)
    ax_graph.set_ylim(0, 100)
    ax_graph.invert_yaxis() # Dibalik agar koordinat (0,0) ada di Kiri-Atas sama seperti piksel gambar
    ax_graph.grid(True, linestyle='--', alpha=0.6)
    ax_graph.set_xlabel("Sumbu Koordinat X (Lebar Wajah)", fontsize=11)
    ax_graph.set_ylabel("Sumbu Koordinat Y (Tinggi Wajah)", fontsize=11)

    # A. Plot Kluster Histori Database (Dot Kecil)
    ax_graph.scatter(db_mata_kiri_x, db_mata_kiri_y, color='#ff7675', s=40, alpha=0.5, label='Kluster Mata Kiri (DB)')
    ax_graph.scatter(db_mata_kanan_x, db_mata_kanan_y, color='#d63031', s=40, alpha=0.5, label='Kluster Mata Kanan (DB)')
    ax_graph.scatter(db_hidung_x, db_hidung_y, color='#74b9ff', s=40, alpha=0.5, label='Kluster Hidung (DB)')
    ax_graph.scatter(db_mulut_x, db_mulut_y, color='#55efc4', s=40, alpha=0.5, label='Kluster Mulut (DB)')

    # B. Plot Nilai Rata-rata Pusat Kluster (Bintang Besar)
    ax_graph.scatter([avg_fitur["mata_kiri"][0]], [avg_fitur["mata_kiri"][1]], color='red', marker='*', s=200, edgecolors='black', linewidths=1.5, label='Rata-Rata (Centroid DB)')
    ax_graph.scatter([avg_fitur["mata_kanan"][0]], [avg_fitur["mata_kanan"][1]], color='red', marker='*', s=200, edgecolors='black', linewidths=1.5)
    ax_graph.scatter([avg_fitur["hidung"][0]], [avg_fitur["hidung"][1]], color='blue', marker='*', s=200, edgecolors='black', linewidths=1.5)
    ax_graph.scatter([avg_fitur["mulut"][0]], [avg_fitur["mulut"][1]], color='green', marker='*', s=200, edgecolors='black', linewidths=1.5)

    # C. Plot Titik Wajah Input Saat Ini (Simbol 'X' Tebal / Berwarna Magenta)
    if len(inp_eyes) == 2:
        ax_graph.scatter([inp_eyes[0][0]], [inp_eyes[0][1]], color='magenta', marker='X', s=150, linewidths=2.5, label='Titik Wajah Input Baru')
        ax_graph.scatter([inp_eyes[1][0]], [inp_eyes[1][1]], color='magenta', marker='X', s=150, linewidths=2.5)
    if inp_nose:
        ax_graph.scatter([inp_nose[0][0]], [inp_nose[0][1]], color='magenta', marker='X', s=150, linewidths=2.5)
    if inp_mouth:
        ax_graph.scatter([inp_mouth[0][0]], [inp_mouth[0][1]], color='magenta', marker='X', s=150, linewidths=2.5)

    # Tambah teks label status kelulusan threshold di dalam grafik
    status_text = "DIKENALI" if hasil_match["dikenali"] else "TIDAK DIKENALI"
    color_box = '#d4edda' if hasil_match["dikenali"] else '#f8d7da'
    color_text = 'green' if hasil_match["dikenali"] else 'red'
    
    props = dict(boxstyle='round,pad=0.8', facecolor=color_box, edgecolor=color_text, alpha=0.9)
    ax_graph.text(5, 92, f"STATUS KEPUTUSAN: {status_text}\nKandidat: {kandidat_nama.upper()}\nSkor Jarak ke DB: {hasil_match['jarak']:.4f}", 
                  color=color_text, fontsize=11, fontweight='bold', bbox=props)

    ax_graph.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    print("\n[SUKSES] Grafik analisis koordinat sebaran titik berhasil ditampilkan!")
    plt.show()

if __name__ == "__main__":
    main()