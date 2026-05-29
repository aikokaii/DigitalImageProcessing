import numpy as np
import json
import os
def fitur_ke_vektor(fitur: dict) -> np.ndarray:
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

def normalisasi_vektor(v: np.ndarray, eps=1e-8) -> np.ndarray:
    """Normalisasi vektor ke rentang [0, 1] (min-max per elemen)."""
    v_min = v.min()
    v_max = v.max()
    return (v - v_min) / (v_max - v_min + eps)

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Hitung jarak Euclidean antara dua vektor fitur.
    d = sqrt( sum( (xi - yi)^2 ) )
    Nilai semakin kecil = semakin mirip.
    """
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))

def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    d = sum( |xi - yi| )
    """
    return float(np.sum(np.abs(v1 - v2)))

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

class FaceMatcher:
    THRESHOLD_DEFAULT = 12.0

    def __init__(self, db_path: str = "database_fitur.json"):
        self.db_path  = db_path
        self.database = {}          
        self._load_database()
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

    def match(self, fitur_input: dict,
              threshold: float = None,
              metode: str = "euclidean") -> dict:
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
            "nama_kandidat": nama_terbaik,   
            "jarak"     : round(skor_terbaik, 4),
            "threshold" : threshold,
            "dikenali"  : dikenali,
            "semua_skor": [(n, round(s, 4)) for n, s in skor_per_orang],
            "metode"    : metode,
        }

    def daftar_profil(self) -> list:
        """Kembalikan daftar nama yang terdaftar di database."""
        return list(self.database.keys())