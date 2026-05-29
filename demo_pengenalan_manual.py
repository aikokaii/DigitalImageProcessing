import os
import json
from src.pencocokan import FaceMatcher

def main():
    print("="*60)
    print("   DEMO FACE RECOGNITION (Membandingkan 2 Database JSON)")
    print("="*60)
    
    # 1. Buka database utama (Dataset)
    db_path = "database_fitur.json"
    if not os.path.exists(db_path):
        print(f"File {db_path} tidak ditemukan! Harap jalankan ekstraksi_fitur.py dulu.")
        return
        
    matcher = FaceMatcher(db_path=db_path)
    
    # 2. Buka database uji (Data Testing)
    db_uji_path = "database_uji.json"
    if not os.path.exists(db_uji_path):
        print(f"File {db_uji_path} tidak ditemukan! Harap jalankan ekstraksi_fitur_uji.py dulu.")
        return
        
    with open(db_uji_path, 'r') as f:
        db_uji = json.load(f)
        
    if not db_uji:
        print(f"Database uji '{db_uji_path}' kosong.")
        return
        
    print(f"\n--- Memulai Uji Coba dari {db_uji_path} ---\n")
    
    count_benar = 0
    total_uji = 0
    
    # Loop untuk semua data di database_uji.json
    for nama_sebenarnya, list_fitur_uji in db_uji.items():
        for i, fitur_uji in enumerate(list_fitur_uji):
            total_uji += 1
            
            # Cocokkan vektor fitur uji dengan database_fitur.json 
            hasil = matcher.match(fitur_uji, threshold=15.0, metode="euclidean")
            
            # Cek kebenaran (apakah tebakan sesuai dengan label aslinya)
            tebakan = hasil['nama_kandidat'].lower()
            is_benar = (tebakan == nama_sebenarnya.lower())
            if is_benar:
                count_benar += 1
                
            print(f"Data Uji    : {nama_sebenarnya.upper()} (Sampel ke-{i+1})")
            if hasil['dikenali']:
                print(f"Prediksi    : {hasil['nama'].upper()} (DIKENALI)")
            else:
                print(f"Prediksi    : TIDAK DIKENALI (Mirip: {hasil['nama_kandidat']})")
            
            print(f"Status      : {'BENAR' if is_benar else 'SALAH'}")
            print(f"Jarak Euclid: {hasil['jarak']:.4f}")
            print("Kandidat Terdekat:")
            for nama, skor in hasil['semua_skor'][:3]: # Tampilkan Top 3
                print(f" -> {nama:<15} (Jarak: {skor:.4f})")
            print("-" * 50)
            
    print(f"\n[REKAPITULASI]")
    print(f"Total Data Uji : {total_uji}")
    print(f"Prediksi Benar : {count_benar}")
    if total_uji > 0:
        print(f"Akurasi        : {(count_benar/total_uji)*100:.2f}%")

if __name__ == "__main__":
    main()
