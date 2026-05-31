import os
import json
from src.pencocokan import FaceMatcher
from eigenface import EigenfaceRecognizer

def main():
    print("="*60)
    print("   KOMPARASI AKURASI: EKSTRAKSI FITUR VS EIGENFACE")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. SETUP PENDEKATAN EKSTRAKSI FITUR (JSON)
    # ---------------------------------------------------------
    db_fitur_path = "database_fitur.json"
    db_uji_path = "database_uji.json"
    
    if not os.path.exists(db_fitur_path) or not os.path.exists(db_uji_path):
        print("Error: database_fitur.json atau database_uji.json tidak ditemukan.")
        return
        
    matcher = FaceMatcher(db_path=db_fitur_path)
    
    with open(db_uji_path, 'r') as f:
        db_uji = json.load(f)
        
    # ---------------------------------------------------------
    # 2. SETUP PENDEKATAN EIGENFACE
    # ---------------------------------------------------------
    folder_latih = "hasil_deteksi"
    folder_uji = "datatest"
    
    if not os.path.exists(folder_latih) or not os.path.exists(folder_uji):
        print("Error: folder hasil_deteksi atau datatest tidak ditemukan.")
        return
        
    ef = EigenfaceRecognizer()
    sukses_ef = ef.train(folder_latih)
    if not sukses_ef:
        print("Error: Pelatihan Eigenface gagal.")
        return

    # ---------------------------------------------------------
    # 3. EVALUASI BERSAMA
    # ---------------------------------------------------------
    print("\n--- Memulai Evaluasi pada Data Uji ---")
    
    total_data = 0
    benar_fitur = 0
    benar_eigen = 0
    
    # Kumpulkan semua data uji dari database_uji
    for nama_sebenarnya, list_fitur in db_uji.items():
        for fitur_uji in list_fitur:
            file_uji = fitur_uji.get("file")
            
            if not file_uji:
                print(f"[SKIP] Data {nama_sebenarnya} tidak memiliki atribut 'file'. Harap ekstrak ulang fitur uji.")
                continue
                
            path_uji = os.path.join(folder_uji, file_uji)
            if not os.path.exists(path_uji):
                print(f"[SKIP] File {path_uji} tidak ditemukan.")
                continue
            
            total_data += 1
            label_asli = nama_sebenarnya.lower()
            kata_label = label_asli.replace('_', ' ').replace('-', ' ').split()
            
            # --- Uji Pendekatan Fitur JSON ---
            hasil_fitur = matcher.match(fitur_uji, threshold=15.0, metode="euclidean")
            tebakan_fitur = hasil_fitur['nama_kandidat'].lower()
            is_benar_fitur = (tebakan_fitur == label_asli) or (tebakan_fitur in kata_label)
            if is_benar_fitur:
                benar_fitur += 1
                
            # --- Uji Pendekatan Eigenface ---
            hasil_ef = ef.predict(path_uji, threshold=None) # Jangan pakai threshold untuk akurasi top-1
            if hasil_ef:
                tebakan_ef = hasil_ef['kandidat_terdekat'].lower()
                is_benar_ef = (tebakan_ef == label_asli) or (tebakan_ef in kata_label)
                if is_benar_ef:
                    benar_eigen += 1
            else:
                tebakan_ef = "gagal"
                is_benar_ef = False
                
            print(f"\n[Data Uji {total_data}] {file_uji} (Label Asli: {label_asli.upper()})")
            print(f"  -> Ekstraksi Fitur : {tebakan_fitur.upper()} ({'BENAR' if is_benar_fitur else 'SALAH'})")
            print(f"  -> Eigenface       : {tebakan_ef.upper()} ({'BENAR' if is_benar_ef else 'SALAH'})")

    # ---------------------------------------------------------
    # 4. REKAPITULASI
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("   KESIMPULAN & REKAPITULASI AKURASI")
    print("="*60)
    print(f"Total Data Uji Diproses : {total_data}")
    
    if total_data > 0:
        akurasi_fitur = (benar_fitur / total_data) * 100
        akurasi_eigen = (benar_eigen / total_data) * 100
        
        print(f"1. Pendekatan Ekstraksi Fitur (Biologis/Grid):")
        print(f"   - Prediksi Benar : {benar_fitur}")
        print(f"   - Akurasi        : {akurasi_fitur:.2f}%\n")
        
        print(f"2. Pendekatan Eigenface (PCA):")
        print(f"   - Prediksi Benar : {benar_eigen}")
        print(f"   - Akurasi        : {akurasi_eigen:.2f}%\n")
        
        if akurasi_fitur > akurasi_eigen:
            print("=> KESIMPULAN: Pendekatan Ekstraksi Fitur lebih akurat pada dataset ini.")
        elif akurasi_eigen > akurasi_fitur:
            print("=> KESIMPULAN: Pendekatan Eigenface lebih akurat pada dataset ini.")
        else:
            print("=> KESIMPULAN: Kedua pendekatan memiliki akurasi yang sama pada dataset ini.")
            
if __name__ == "__main__":
    main()
