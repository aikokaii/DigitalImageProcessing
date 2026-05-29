"""
src/pencocokan.py
-----------------
Modul Pencocokan Wajah (Feature Matching) + Identifikasi — Tahap 6 & 7.
Membandingkan vektor fitur wajah input dengan semua profil di database
menggunakan Euclidean Distance, lalu menentukan identitas berdasarkan threshold.

MURNI MANUAL — tidak menggunakan OpenCV maupun library ML.
"""

import numpy as np
import json
import os


# ============================================================
# HELPER: Ekstrak vektor numerik dari dict fitur
# ============================================================

def fitur_ke_vektor(fitur: dict) -> np.ndarray:
    """
    Ubah dict fitur (hasil extract_features) menjadi vektor numpy 1D.
    Hanya ambil nilai numerik yang relevan untuk perbandingan —
    koordinat (eye_coords, nose_coords, mouth_coords) dilewati
    karena bergantung pada posisi crop, bukan identitas wajah.
    """
    vektor = []

    # --- Fitur skalar geometris & biologis ---
    kunci_skalar = [
        "eye_distance",
        "eye_to_nose_ratio",
        "nose_to_mouth_ratio",
        "skin_ratio",
        "edge_density",
        "contour_density",
        "eye_region_edge",
        "eye_non_skin",
        "nose_region_edge",
        "mouth_non_skin",
        "symmetry_error",
    ]
    for k in kunci_skalar:
        vektor.append(float(fitur.get(k, 0.0)))

    # --- Fitur vektor grid & tekstur ---
    for k in ["grid_edge_density", "grid_skin_ratio", "grid_intensity", "lbp_histogram"]:
        vals = fitur.get(k, [])
        vektor.extend([float(v) for v in vals])

    return np.array(vektor, dtype=np.float64)


# ============================================================
# NORMALISASI: Z-score sederhana agar skala tiap fitur setara
# ============================================================

def normalisasi_vektor(v: np.ndarray, eps=1e-8) -> np.ndarray:
    """Normalisasi vektor ke rentang [0, 1] (min-max per elemen)."""
    v_min = v.min()
    v_max = v.max()
    return (v - v_min) / (v_max - v_min + eps)


# ============================================================
# JARAK: Euclidean Distance (Tahap 6a dari workflow)
# ============================================================

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Hitung jarak Euclidean antara dua vektor fitur.
    d = sqrt( sum( (xi - yi)^2 ) )
    Nilai semakin kecil = semakin mirip.
    """
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


# ============================================================
# JARAK: Manhattan Distance (Tahap 6b dari workflow)
# ============================================================

def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    d = sum( |xi - yi| )
    """
    return float(np.sum(np.abs(v1 - v2)))


# ============================================================
# SIMILARITY: Cosine Similarity (Tahap 6c dari workflow)
# ============================================================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Ukur kemiripan arah vektor. Nilai mendekati 1 = sangat mirip.
    sim = (v1 · v2) / (||v1|| * ||v2||)
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ============================================================
# KELAS UTAMA: FaceMatcher
# ============================================================

class FaceMatcher:
    """
    Memuat database fitur wajah dan melakukan pencocokan
    terhadap fitur wajah input baru.

    Cara kerja:
    1. Load database_fitur.json
    2. Untuk setiap profil di database, hitung jarak ke fitur input
    3. Ambil profil dengan jarak terkecil
    4. Bandingkan dengan threshold — putuskan DIKENALI / TIDAK DIKENALI
    """

    # Threshold default — jarak Euclidean di bawah nilai ini = dikenali.
    # Nilai ini bisa disesuaikan lewat parameter threshold saat memanggil match().
    THRESHOLD_DEFAULT = 12.0

    def __init__(self, db_path: str = "database_fitur.json"):
        self.db_path  = db_path
        self.database = {}          # { nama: [vektor_fitur, ...] }
        self._load_database()

    # ----------------------------------------------------------
    # Load & parse database
    # ----------------------------------------------------------

    def _load_database(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database '{self.db_path}' tidak ditemukan!\n"
                "Jalankan ekstraksi_fitur.py terlebih dahulu."
            )
        with open(self.db_path, 'r') as f:
            raw = json.load(f)

        for nama, list_fitur in raw.items():
            vektors = []
            for fitur_dict in list_fitur:
                v = fitur_ke_vektor(fitur_dict)
                vektors.append(v)
            self.database[nama] = vektors

        total_profil = sum(len(v) for v in self.database.values())
        print(f"[Database] {len(self.database)} orang, {total_profil} total entri fitur dimuat.")

    # ----------------------------------------------------------
    # Pencocokan utama
    # ----------------------------------------------------------

    def match(self, fitur_input: dict,
              threshold: float = None,
              metode: str = "euclidean") -> dict:
        """
        Cocokkan fitur wajah input ke database.

        Parameter:
            fitur_input : dict hasil extract_features()["features"]
            threshold   : batas jarak — lebih kecil = lebih ketat
                          (default 12.0 untuk euclidean, 0.85 untuk cosine)
            metode      : "euclidean" | "manhattan" | "cosine"

        Return dict:
            {
                "nama"       : str   — nama yang paling cocok (atau "TIDAK DIKENALI"),
                "jarak"      : float — nilai jarak/similarity terbaik,
                "threshold"  : float — threshold yang digunakan,
                "dikenali"   : bool  — True jika di bawah threshold,
                "semua_skor" : list of (nama, jarak) urut dari terbaik,
                "metode"     : str
            }
        """
        if not self.database:
            return {"nama": "TIDAK DIKENALI", "jarak": float('inf'),
                    "dikenali": False, "semua_skor": [], "metode": metode,
                    "threshold": threshold}

        # Tentukan threshold default per metode
        if threshold is None:
            if metode == "cosine":
                threshold = 0.85   # similarity — harus DI ATAS threshold
            elif metode == "manhattan":
                threshold = 50.0
            else:
                threshold = self.THRESHOLD_DEFAULT

        v_input = normalisasi_vektor(fitur_ke_vektor(fitur_input))

        skor_per_orang = []

        for nama, list_vektor in self.database.items():
            skor_terbaik_orang = None

            for v_db in list_vektor:
                v_db_norm = normalisasi_vektor(v_db)

                if metode == "euclidean":
                    skor = euclidean_distance(v_input, v_db_norm)
                    # Untuk euclidean/manhattan: skor kecil = lebih mirip
                    if skor_terbaik_orang is None or skor < skor_terbaik_orang:
                        skor_terbaik_orang = skor

                elif metode == "manhattan":
                    skor = manhattan_distance(v_input, v_db_norm)
                    if skor_terbaik_orang is None or skor < skor_terbaik_orang:
                        skor_terbaik_orang = skor

                elif metode == "cosine":
                    skor = cosine_similarity(v_input, v_db_norm)
                    # Untuk cosine: skor besar = lebih mirip
                    if skor_terbaik_orang is None or skor > skor_terbaik_orang:
                        skor_terbaik_orang = skor

            if skor_terbaik_orang is not None:
                skor_per_orang.append((nama, skor_terbaik_orang))

        # Urutkan: euclidean/manhattan ascending, cosine descending
        if metode == "cosine":
            skor_per_orang.sort(key=lambda x: x[1], reverse=True)
            nama_terbaik, skor_terbaik = skor_per_orang[0]
            dikenali = skor_terbaik >= threshold
        else:
            skor_per_orang.sort(key=lambda x: x[1])
            nama_terbaik, skor_terbaik = skor_per_orang[0]
            dikenali = skor_terbaik <= threshold

        return {
            "nama"      : nama_terbaik if dikenali else "TIDAK DIKENALI",
            "nama_kandidat": nama_terbaik,   # kandidat terbaik walau tidak lolos threshold
            "jarak"     : round(skor_terbaik, 4),
            "threshold" : threshold,
            "dikenali"  : dikenali,
            "semua_skor": [(n, round(s, 4)) for n, s in skor_per_orang],
            "metode"    : metode,
        }

    # ----------------------------------------------------------
    # Info database
    # ----------------------------------------------------------

    def daftar_profil(self) -> list:
        """Kembalikan daftar nama yang terdaftar di database."""
        return list(self.database.keys())