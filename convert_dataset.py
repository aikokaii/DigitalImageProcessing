import os
from PIL import Image

def process_dataset(folder_path, max_images=150):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' tidak ditemukan.")
        return

    # Kumpulkan semua file dalam folder
    files = os.listdir(folder_path)
    
    # 1. Hapus semua file yang mengandung kata "eye" atau ekstensi .eye
    eye_files = [f for f in files if 'eye' in f.lower() or f.lower().endswith('.eye')]
    for eye_file in eye_files:
        try:
            os.remove(os.path.join(folder_path, eye_file))
            print(f"Dihapus (file eye): {eye_file}")
        except Exception as e:
            print(f"Gagal menghapus {eye_file}: {e}")

    # Update daftar file setelah penghapusan
    files = os.listdir(folder_path)
    
    # 2. Cari semua file .pgm
    pgm_files = sorted([f for f in files if f.lower().endswith('.pgm')])
    
    # Batasi hanya 100 gambar pertama
    pgm_to_process = pgm_files[:max_images]
    
    # Hapus sisa file .pgm yang melebihi batas 100 (opsional, tapi disarankan agar rapi)
    # Jika user hanya ingin memproses 100 dan sisa pgm dibiarkan, kita skip penghapusan sisanya.
    # Disini kita biarkan saja sisa pgm, atau kita bisa hapus semuanya.
    # Berdasarkan instruksi: "convert 100 gambar pgm pertama saja... juga pgm nya jaddi abis convert hapus"
    # Kita hanya menghapus pgm yang dikonversi.
    
    print(f"\nDitemukan {len(pgm_files)} file PGM. Mengonversi {len(pgm_to_process)} file pertama ke JPG...")
    
    count = 0
    for pgm_file in pgm_to_process:
        pgm_path = os.path.join(folder_path, pgm_file)
        jpg_filename = os.path.splitext(pgm_file)[0] + ".jpg"
        jpg_path = os.path.join(folder_path, jpg_filename)
        
        try:
            # Buka dan konversi ke RGB
            img = Image.open(pgm_path)
            img.load() # Paksa baca ke memori
            rgb_img = img.convert('RGB')
            img.close() # Tutup file secara eksplisit
            
            rgb_img.save(jpg_path, 'JPEG')
            
            # Hapus file pgm aslinya setelah berhasil diconvert
            os.remove(pgm_path)
            print(f"[{count+1}/{len(pgm_to_process)}] Dikonversi & dihapus: {pgm_file} -> {jpg_filename}")
            count += 1
            
        except Exception as e:
            print(f"Gagal memproses {pgm_file}: {e}")
            
    print(f"\nSelesai! {count} file .pgm berhasil dikonversi ke .jpg dan file aslinya dihapus.")

if __name__ == "__main__":
    folder_dataset = "dataset" # Nama folder disesuaikan dengan permintaan
    process_dataset(folder_dataset, max_images=100)
