"""
demo_pengenalan_manual.py
=========================
Uji batch: cocokkan semua data di database_uji.json ke database_fitur.json.
Versi TANPA threshold -- selalu tampilkan prediksi rank-1.
Threshold akan dikalibrasi setelah dataset lengkap (30 foto/orang).
"""

import os
import json

DB_LATIH = "database_fitur.json"
DB_UJI   = "database_uji.json"
METODE   = "manhattan"   # "euclidean" | "manhattan" | "cosine"

# Import fungsi hitung skor langsung (bypass layer threshold)
from src.pencocokan import fitur_ke_vektor, normalisasi_vektor
from src.pencocokan import euclidean_distance, manhattan_distance, cosine_similarity


def hitung_semua_skor(v_input, database, metode, test_filename=""):
    """Kembalikan list (nama, skor_rata) diurutkan dari paling mirip."""
    hasil = []
    for nama, list_data in database.items():
        skor_list = []
        for file_db, v_db in list_data:
            if test_filename and file_db == test_filename:
                continue
            v_db_n = normalisasi_vektor(v_db)
            if metode == "euclidean":
                skor_list.append(euclidean_distance(v_input, v_db_n))
            elif metode == "manhattan":
                skor_list.append(manhattan_distance(v_input, v_db_n))
            elif metode == "cosine":
                skor_list.append(cosine_similarity(v_input, v_db_n))
        if skor_list:
            if metode == "cosine":
                hasil.append((nama, max(skor_list)))
            else:
                hasil.append((nama, min(skor_list)))

    reverse = (metode == "cosine")
    hasil.sort(key=lambda x: x[1], reverse=reverse)
    return hasil


def main():
    # --- Validasi file ---
    for path in [DB_LATIH, DB_UJI]:
        if not os.path.exists(path):
            print("[ERROR] File '%s' tidak ditemukan!" % path)
            return

    # --- Muat database latih ---
    with open(DB_LATIH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    database = {}
    for nama, list_fitur in raw.items():
        database[nama] = []
        for fd in list_fitur:
            file_name = fd.get("file", "")
            path_asli = os.path.join("hasil_deteksi", file_name)
            if os.path.exists(path_asli):
                database[nama].append((file_name, fitur_ke_vektor(fd)))
    total_sampel = sum(len(v) for v in database.values())
    print("[Database] %d subjek, %d total sampel." % (len(database), total_sampel))
    print("Subjek terdaftar: %s" % list(database.keys()))

    # --- Muat database uji ---
    with open(DB_UJI, "r", encoding="utf-8") as f:
        db_uji = json.load(f)

    if not db_uji:
        print("[ERROR] '%s' kosong." % DB_UJI)
        return

    total_uji = sum(len(v) for v in db_uji.values())

    print()
    print("=" * 60)
    print("  FACE RECOGNITION -- UJI BATCH (Tanpa Threshold)")
    print("  Metode : %s" % METODE.upper())
    print("=" * 60)

    count_benar = 0
    i_global    = 0

    for nama_asli, list_fitur_uji in db_uji.items():
        for no, fitur_uji in enumerate(list_fitur_uji, start=1):
            i_global += 1

            v_input    = normalisasi_vektor(fitur_ke_vektor(fitur_uji))
            test_filename = fitur_uji.get("file", "")
            skor_semua = hitung_semua_skor(v_input, database, METODE, test_filename)

            if not skor_semua:
                print("[%d/%d] %s #%d: Database kosong" % (i_global, total_uji, nama_asli, no))
                continue

            prediksi, skor_terbaik = skor_semua[0]

            # Bersihkan nama asli dari kata 'datatest'
            nama_asli_bersih = nama_asli.lower().replace("datatest", "").strip()
            prediksi_bersih = prediksi.lower().strip()

            is_benar = prediksi_bersih == nama_asli_bersih
            if is_benar:
                count_benar += 1

            status = "BENAR" if is_benar else "SALAH"

            print()
            print("[%d/%d] Data Uji: %s (sampel ke-%d)" % (
                i_global, total_uji, nama_asli.upper(), no))
            print("  " + "-" * 50)
            print("  Prediksi  : %s" % prediksi.upper())
            print("  Skor      : %.4f" % skor_terbaik)
            print("  Status    : [%s] %s" % ("OK" if is_benar else "XX", status))
            print("  Top 3 kandidat:")
            for rank, (nama, skor) in enumerate(skor_semua[:3], 1):
                marker = " <--" if rank == 1 else ""
                print("    %d. %-20s skor=%.4f%s" % (rank, nama, skor, marker))

    # --- Rekapitulasi ---
    print()
    print("=" * 60)
    print("  REKAPITULASI")
    print("=" * 60)
    print("  Total Data Uji     : %d" % total_uji)
    print("  Prediksi Benar     : %d" % count_benar)
    print("  Prediksi Salah     : %d" % (total_uji - count_benar))
    if total_uji > 0:
        print("  Akurasi            : %.2f%%" % (count_benar / total_uji * 100))
    print("=" * 60)


if __name__ == "__main__":
    main()
