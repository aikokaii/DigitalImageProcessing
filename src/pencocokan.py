"""
src/pencocokan.py
=================
Modul pencocokan wajah dengan sistem thresholding komprehensif.

Sistem threshold terdiri dari 3 lapis:
  1. ABSOLUTE THRESHOLD  — skor terbaik harus memenuhi batas minimum
  2. MARGIN THRESHOLD    — selisih rank-1 dan rank-2 harus cukup besar
                           (mencegah ambiguitas antar subjek yang mirip)
  3. CONFIDENCE LEVEL    — menggabungkan keduanya menjadi tingkat keyakinan
                           (HIGH / MEDIUM / LOW / REJECTED)

Dengan 30 foto/subjek, sistem mengambil rata-rata skor dari semua sampel
(lebih robust dari sekedar skor terbaik).
"""

import numpy as np
import json
import os
import math


# ==============================================================
# KONVERSI DICT FITUR → VEKTOR NUMPY
# ==============================================================

KUNCI_SKALAR = [
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

KUNCI_VEKTOR = [
    "grid_edge_density",
    "grid_skin_ratio",
    "grid_intensity",
    "lbp_histogram",
]


def fitur_ke_vektor(fitur: dict) -> np.ndarray:
    vektor = [float(fitur.get(k, 0.0)) for k in KUNCI_SKALAR]
    for k in KUNCI_VEKTOR:
        vektor.extend([float(v) for v in fitur.get(k, [])])
    return np.array(vektor, dtype=np.float64)


def normalisasi_vektor(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Norm-2 normalization (supaya scale invariant untuk metode euclidean/manhattan)."""
    norm = np.linalg.norm(v)
    return v / (norm + eps)


# ==============================================================
# FUNGSI JARAK
# ==============================================================

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.sum(np.abs(v1 - v2)))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ==============================================================
# KONFIGURASI THRESHOLD PER METODE
# ==============================================================
#
# Setiap metode punya 3 parameter:
#   absolute  — batas keras skor rank-1 (reject jika tidak lolos)
#   margin    — selisih minimum rank-1 vs rank-2 (dalam satuan skor)
#   conf_high — batas skor + margin untuk level HIGH
#
# Catatan satuan:
#   euclidean / manhattan : skor KECIL = mirip  (distance)
#   cosine                : skor BESAR = mirip  (similarity)

THRESHOLD_CONFIG = {
    "euclidean": {
        "absolute"  : 0.35,   # rank-1 distance harus <= ini
        "margin_abs": 0.05,   # selisih rank1 vs rank2 harus >= ini
        "margin_rel": 0.15,   # selisih relatif (margin/rank1) harus >= ini
        "conf_high" : 0.20,   # distance <= ini → HIGH
        "conf_medium": 0.30,  # distance <= ini → MEDIUM (sisanya LOW)
    },
    "manhattan": {
        "absolute"  : 10.0,
        "margin_abs": 1.0,
        "margin_rel": 0.10,
        "conf_high" : 5.0,
        "conf_medium": 8.0,
    },
    "cosine": {
        "absolute"  : 0.80,   # similarity harus >= ini
        "margin_abs": 0.03,   # selisih rank1 vs rank2 harus >= ini
        "margin_rel": 0.05,
        "conf_high" : 0.92,   # similarity >= ini → HIGH
        "conf_medium": 0.85,
    },
}


# ==============================================================
# KELAS HASIL PENCOCOKAN
# ==============================================================

class HasilPencocokan:
    """Hasil akhir pencocokan satu gambar uji."""

    def __init__(self, nama_prediksi: str, dikenali: bool,
                 skor_terbaik: float, confidence: str,
                 alasan_reject: str, semua_skor: list,
                 metode: str, threshold_cfg: dict):
        self.nama_prediksi  = nama_prediksi    # nama subjek atau "TIDAK DIKENALI"
        self.dikenali       = dikenali
        self.skor_terbaik   = skor_terbaik
        self.confidence     = confidence       # "HIGH" / "MEDIUM" / "LOW" / "REJECTED"
        self.alasan_reject  = alasan_reject    # "" jika dikenali
        self.semua_skor     = semua_skor       # list (nama, skor) semua kandidat
        self.metode         = metode
        self.threshold_cfg  = threshold_cfg

    def __repr__(self):
        return (f"HasilPencocokan(nama={self.nama_prediksi!r}, "
                f"dikenali={self.dikenali}, conf={self.confidence}, "
                f"skor={self.skor_terbaik:.4f})")


# ==============================================================
# KELAS UTAMA: FaceMatcher
# ==============================================================

class FaceMatcher:
    """
    Pencocokan wajah berbasis fitur dengan thresholding berlapis.

    Alur kerja:
      1. Hitung rata-rata skor tiap subjek dari semua sampelnya.
         (dengan 30 foto/subjek, ini jauh lebih stabil dari skor terbaik)
      2. Lapis 1 — Absolute threshold: skor rank-1 harus memenuhi batas.
      3. Lapis 2 — Margin threshold: selisih rank-1 vs rank-2 harus cukup.
      4. Tentukan confidence level dari gabungan skor + margin.
    """

    def __init__(self, db_path: str = "database_fitur.json"):
        self.db_path  = db_path
        self.database = {}
        self._muat_database()

    # ----------------------------------------------------------
    def _muat_database(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database '{self.db_path}' tidak ditemukan!\n"
                "Jalankan ekstraksi_fitur.py terlebih dahulu."
            )
        with open(self.db_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for nama, list_fitur in raw.items():
            self.database[nama] = [
                fitur_ke_vektor(fd) for fd in list_fitur
            ]

        total = sum(len(v) for v in self.database.values())
        print(f"[Database] {len(self.database)} subjek, "
              f"{total} total sampel dimuat dari '{self.db_path}'.")

    # ----------------------------------------------------------
    def daftar_profil(self) -> list:
        return list(self.database.keys())

    # ----------------------------------------------------------
    def _hitung_skor_per_orang(self, v_input: np.ndarray,
                                metode: str) -> list:
        """
        Kembalikan list (nama, skor_rata_rata) untuk semua subjek,
        diurutkan dari yang paling cocok.

        Menggunakan rata-rata dari seluruh sampel subjek → lebih robust
        saat dataset besar (misal 30 foto/subjek).
        """
        hasil = []
        for nama, list_vektor in self.database.items():
            skor_semua = []
            for v_db in list_vektor:
                v_db_n = normalisasi_vektor(v_db)
                if metode == "euclidean":
                    skor_semua.append(euclidean_distance(v_input, v_db_n))
                elif metode == "manhattan":
                    skor_semua.append(manhattan_distance(v_input, v_db_n))
                elif metode == "cosine":
                    skor_semua.append(cosine_similarity(v_input, v_db_n))

            if skor_semua:
                rata = sum(skor_semua) / len(skor_semua)
                hasil.append((nama, rata))

        # Urutkan: jarak kecil dulu (euclidean/manhattan),
        #          similarity besar dulu (cosine)
        reverse = (metode == "cosine")
        hasil.sort(key=lambda x: x[1], reverse=reverse)
        return hasil

    # ----------------------------------------------------------
    def _tentukan_confidence(self, skor: float, margin: float,
                              cfg: dict, metode: str) -> str:
        """
        Tentukan level kepercayaan dari skor rank-1 dan margin rank1–rank2.

        Returns: "HIGH" | "MEDIUM" | "LOW"
        (Dipanggil hanya jika kedua threshold sudah lolos)
        """
        if metode == "cosine":
            # Skor besar = lebih baik
            skor_ok_high = skor >= cfg["conf_high"]
        else:
            # Skor kecil = lebih baik
            skor_ok_high = skor <= cfg["conf_high"]

        margin_ok = (margin >= cfg["margin_abs"] * 2)  # margin besar = HIGH

        if skor_ok_high and margin_ok:
            return "HIGH"

        if metode == "cosine":
            skor_ok_med = skor >= cfg["conf_medium"]
        else:
            skor_ok_med = skor <= cfg["conf_medium"]

        if skor_ok_med:
            return "MEDIUM"

        return "LOW"

    # ----------------------------------------------------------
    def match(self, fitur_input: dict,
              metode: str = "euclidean",
              threshold_override: dict = None) -> HasilPencocokan:
        """
        Cocokkan satu fitur uji ke database.

        Args:
            fitur_input       : dict fitur dari ekstraksi
            metode            : "euclidean" | "manhattan" | "cosine"
            threshold_override: dict opsional untuk override nilai threshold
                                {"absolute": ..., "margin_abs": ..., ...}

        Returns:
            HasilPencocokan
        """
        if not self.database:
            return HasilPencocokan(
                nama_prediksi="TIDAK DIKENALI",
                dikenali=False,
                skor_terbaik=float("inf"),
                confidence="REJECTED",
                alasan_reject="Database kosong",
                semua_skor=[],
                metode=metode,
                threshold_cfg={},
            )

        # Ambil konfigurasi threshold
        cfg = dict(THRESHOLD_CONFIG.get(metode, THRESHOLD_CONFIG["euclidean"]))
        if threshold_override:
            cfg.update(threshold_override)

        # Normalisasi input
        v_input = normalisasi_vektor(fitur_ke_vektor(fitur_input))

        # Hitung skor semua subjek (rata-rata dari semua sampelnya)
        skor_per_orang = self._hitung_skor_per_orang(v_input, metode)

        if not skor_per_orang:
            return HasilPencocokan(
                nama_prediksi="TIDAK DIKENALI",
                dikenali=False,
                skor_terbaik=float("inf"),
                confidence="REJECTED",
                alasan_reject="Tidak ada profil di database",
                semua_skor=[],
                metode=metode,
                threshold_cfg=cfg,
            )

        nama_rank1, skor_rank1 = skor_per_orang[0]

        # Margin antara rank-1 dan rank-2
        if len(skor_per_orang) >= 2:
            _, skor_rank2 = skor_per_orang[1]
            if metode == "cosine":
                margin_abs = skor_rank1 - skor_rank2
                margin_rel = margin_abs / max(skor_rank2, 1e-8)
            else:
                margin_abs = skor_rank2 - skor_rank1
                margin_rel = margin_abs / max(skor_rank1, 1e-8)
        else:
            # Hanya 1 subjek di database
            margin_abs = float("inf")
            margin_rel = float("inf")

        # ===================================================
        # LAPIS 1: Absolute threshold
        # ===================================================
        if metode == "cosine":
            lulus_absolute = skor_rank1 >= cfg["absolute"]
        else:
            lulus_absolute = skor_rank1 <= cfg["absolute"]

        if not lulus_absolute:
            return HasilPencocokan(
                nama_prediksi="TIDAK DIKENALI",
                dikenali=False,
                skor_terbaik=round(skor_rank1, 4),
                confidence="REJECTED",
                alasan_reject=(
                    f"Skor {skor_rank1:.4f} tidak memenuhi batas minimum "
                    f"({'>='+str(cfg['absolute']) if metode=='cosine' else '<='+str(cfg['absolute'])})"
                ),
                semua_skor=[(n, round(s, 4)) for n, s in skor_per_orang],
                metode=metode,
                threshold_cfg=cfg,
            )

        # ===================================================
        # LAPIS 2: Margin threshold
        # ===================================================
        lulus_margin = (margin_abs >= cfg["margin_abs"] or
                        margin_rel >= cfg["margin_rel"])

        if not lulus_margin:
            return HasilPencocokan(
                nama_prediksi="TIDAK DIKENALI",
                dikenali=False,
                skor_terbaik=round(skor_rank1, 4),
                confidence="REJECTED",
                alasan_reject=(
                    f"Margin rank-1 vs rank-2 terlalu kecil "
                    f"(margin={margin_abs:.4f}, min={cfg['margin_abs']}). "
                    f"Prediksi ambigu antara '{nama_rank1}' "
                    f"dan '{skor_per_orang[1][0]}'."
                ),
                semua_skor=[(n, round(s, 4)) for n, s in skor_per_orang],
                metode=metode,
                threshold_cfg=cfg,
            )

        # ===================================================
        # LAPIS 3: Confidence level
        # ===================================================
        confidence = self._tentukan_confidence(skor_rank1, margin_abs, cfg, metode)

        return HasilPencocokan(
            nama_prediksi=nama_rank1,
            dikenali=True,
            skor_terbaik=round(skor_rank1, 4),
            confidence=confidence,
            alasan_reject="",
            semua_skor=[(n, round(s, 4)) for n, s in skor_per_orang],
            metode=metode,
            threshold_cfg=cfg,
        )