import os

def clean_folder():
    folder = "hasil_deteksi"
    if not os.path.exists(folder):
        print(f"Folder '{folder}' tidak ditemukan.")
        return

    count = 0
    for filename in os.listdir(folder):
        name_lower = filename.lower()
        if "skin" in name_lower or "edge" in name_lower or "contour" in name_lower:
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
                print(f"Dihapus: {filename}")
                count += 1
            except Exception as e:
                print(f"Gagal menghapus {filename}: {e}")
                
    print(f"\nSelesai! Berhasil menghapus {count} file (skin/edge/contour) dari '{folder}'.")

if __name__ == "__main__":
    clean_folder()
