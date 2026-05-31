import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import shutil
import json
import subprocess
from PIL import Image, ImageTk

# Import modul pencocokan
from src.pencocokan import fitur_ke_vektor, normalisasi_vektor, manhattan_distance

class FaceRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x700")
        
        self.current_images = []
        self.current_index = 0
        self.results = []
        
        # Load database latih
        self.db_latih = self.load_database("database_fitur.json")

        self.create_widgets()

    def load_database(self, filename):
        if not os.path.exists(filename):
            return {}
        try:
            with open(filename, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            db = {}
            for nama, list_fitur in raw.items():
                db[nama] = []
                for fd in list_fitur:
                    file_name = fd.get("file", "")
                    # PENTING: Hanya muat fitur jika file aslinya masih ada di hasil_deteksi.
                    # Jika file sudah di-move ke datatest, maka tidak akan ikut dilatih.
                    path_asli = os.path.join("hasil_deteksi", file_name)
                    if os.path.exists(path_asli):
                        db[nama].append((file_name, normalisasi_vektor(fitur_ke_vektor(fd))))
                        
            return db
        except Exception as e:
            print(f"Gagal load database {filename}: {e}")
            return {}

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 11), padding=6)
        
        # --- HEADER ---
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=15)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="Sistem Pengenalan Wajah", 
                               font=("Helvetica", 18, "bold"), bg="#2c3e50", fg="white")
        title_label.pack()

        # --- CONTROL PANEL ---
        control_frame = tk.Frame(self.root, pady=20)
        control_frame.pack(fill=tk.X)

        btn_detect = ttk.Button(control_frame, text="1. Deteksi Wajah", command=self.run_detection)
        btn_detect.pack(side=tk.LEFT, padx=10)

        btn_extract = ttk.Button(control_frame, text="2. Ekstrak Fitur Dataset", command=self.run_extraction)
        btn_extract.pack(side=tk.LEFT, padx=10)

        btn_test = ttk.Button(control_frame, text="3. Pilih & Uji Data", command=self.select_and_extract)
        btn_test.pack(side=tk.LEFT, padx=10)
        
        # --- DISPLAY PANEL ---
        self.display_frame = tk.Frame(self.root, bg="#ecf0f1", relief=tk.SUNKEN, bd=2)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.lbl_image = tk.Label(self.display_frame, bg="#ecf0f1")
        self.lbl_image.pack(pady=15)
        
        self.lbl_result = tk.Label(self.display_frame, text="", font=("Helvetica", 16, "bold"), bg="#ecf0f1", fg="#27ae60")
        self.lbl_result.pack(pady=5)
        
        self.lbl_filename = tk.Label(self.display_frame, text="", font=("Helvetica", 10), bg="#ecf0f1", fg="#7f8c8d")
        self.lbl_filename.pack(pady=5)

        # --- PAGINATION PANEL ---
        nav_frame = tk.Frame(self.root, pady=10)
        nav_frame.pack(fill=tk.X)
        
        self.btn_prev = ttk.Button(nav_frame, text="<< Prev", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=20)
        
        self.lbl_page = tk.Label(nav_frame, text="0 / 0", font=("Helvetica", 11))
        self.lbl_page.pack(side=tk.LEFT, expand=True)
        
        self.btn_next = ttk.Button(nav_frame, text="Next >>", command=self.next_image, state=tk.DISABLED)
        self.btn_next.pack(side=tk.RIGHT, padx=20)

    def run_detection(self):
        try:
            messagebox.showinfo("Info", "Menjalankan deteksi_wajah_cepat.py...\nIni mungkin butuh waktu beberapa detik.")
            subprocess.Popen(["python", "deteksi_wajah_cepat.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menjalankan deteksi: {e}")

    def run_extraction(self):
        try:
            messagebox.showinfo("Proses", "Mengekstrak fitur dataset latih (database_fitur.json)...\nProses ini bisa memakan waktu hingga beberapa menit. Harap tunggu.")
            subprocess.run(["python", "ekstraksi_fitur_cepat.py"], check=True)
            self.db_latih = self.load_database("database_fitur.json")
            messagebox.showinfo("Selesai", "Ekstraksi fitur dataset latih selesai!")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengekstrak fitur: {e}")

    def select_and_extract(self):
        if not self.db_latih:
            self.db_latih = self.load_database("database_fitur.json")
            if not self.db_latih:
                messagebox.showwarning("Peringatan", "Database Latih (database_fitur.json) kosong! Ekstrak fitur training terlebih dahulu.")
                return

        filetypes = (
            ('Image files', '*.png *.jpg *.jpeg'),
            ('All files', '*.*')
        )
        
        init_dir = os.path.join(os.getcwd(), "hasil_deteksi")
        if not os.path.exists(init_dir):
            init_dir = os.getcwd()
            
        filenames = filedialog.askopenfilenames(
            title='Pilih Gambar Untuk Uji',
            initialdir=init_dir,
            filetypes=filetypes
        )

        if not filenames:
            return

        # 1. Pindah File ke folder datatest
        target_dir = os.path.join(os.getcwd(), "datatest")
        os.makedirs(target_dir, exist_ok=True)
        
        # Kosongkan datatest terlebih dahulu agar hanya memproses yang baru diuji
        for f in os.listdir(target_dir):
            try:
                os.remove(os.path.join(target_dir, f))
            except:
                pass
        
        self.current_images = []
        
        for f in filenames:
            basename = os.path.basename(f)
            new_path = os.path.join(target_dir, basename)
            try:
                # MOVE file, not copy, so it's separated from training data
                shutil.move(f, new_path)
                self.current_images.append(new_path)
            except Exception as e:
                print(f"Gagal move {f}: {e}")
                
        if not self.current_images:
            return
            
        # 2. Jalankan Ekstraksi Fitur Uji secara sinkron (agar selesai sebelum pindah ke langkah 3)
        messagebox.showinfo("Memproses", f"Mengekstrak fitur untuk {len(self.current_images)} gambar...\nHarap tunggu.")
        try:
            # Gunakan subprocess.run agar UI menunggu sampai script selesai
            subprocess.run(["python", "ekstraksi_fitur_uji.py"], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menjalankan ekstraksi fitur uji: {e}")
            return
            
        # PENTING: Muat ulang database latih agar file yang baru saja di-move ke datatest
        # benar-benar terfilter dan tidak ikut dihitung (memastikan terpisah).
        self.db_latih = self.load_database("database_fitur.json")
            
        # 3. Lakukan Recognition membaca dari database_uji.json
        self.perform_recognition()

    def perform_recognition(self):
        # Muat database uji yang baru saja di-generate
        db_uji = self.load_database("database_uji.json")
        if not db_uji:
            messagebox.showerror("Error", "Gagal memuat database_uji.json. Ekstraksi mungkin gagal.")
            return

        self.results = []
        
        # database_uji.json memiliki struktur { "nama_asli": [fitur1, fitur2, ...] }
        # Tapi yang kita butuhkan adalah mencocokkan setiap file ke database latih
        # Untuk UI, kita asumsikan urutannya sama atau kita baca isi mentah JSON
        try:
            with open("database_uji.json", "r", encoding="utf-8") as f:
                raw_uji = json.load(f)
                
            # Flatten data uji (gabungkan jika ada banyak array menjadi satu list file)
            semua_fitur_uji = []
            for list_fitur in raw_uji.values():
                for fd in list_fitur:
                    semua_fitur_uji.append(fd)
                    
            # Sesuaikan urutan current_images dengan fitur hasil (berdasarkan nama file)
            # Karena di ekstraksi_fitur_uji.py, fitur_dict ditambahkan kunci "file"
            fitur_map = {fd.get("file"): fd for fd in semua_fitur_uji}
            
            for img_path in self.current_images:
                basename = os.path.basename(img_path)
                fd = fitur_map.get(basename)
                
                if not fd:
                    self.results.append({"nama": "GAGAL EKSTRAK", "jarak": 99.9, "conf": 0.0})
                    continue
                    
                v_input = normalisasi_vektor(fitur_ke_vektor(fd))
                
                terbaik_nama = "TIDAK DIKENALI"
                terbaik_jarak = float('inf')
                
                # Nama file yang sedang diuji
                uji_basename = os.path.basename(img_path)
                
                for nama_latih, list_data_db in self.db_latih.items():
                    for file_latih, v_db in list_data_db:
                        # FILTER: Jangan cocokkan dengan diri sendiri jika belum terhapus dari json latih
                        if file_latih == uji_basename:
                            continue
                            
                        dist = manhattan_distance(v_input, v_db)
                        if dist < terbaik_jarak:
                            terbaik_jarak = dist
                            terbaik_nama = nama_latih
                            
                # Konversi ke persentase (Manhattan: 0=100%, 1.0=0%)
                conf = max(0.0, min(100.0, 100.0 - (terbaik_jarak * 100)))
                
                self.results.append({
                    "nama": terbaik_nama.upper(),
                    "jarak": terbaik_jarak,
                    "conf": conf
                })
                
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat mencocokkan fitur: {e}")
            return

        # 4. Tampilkan Hasil
        if self.results:
            self.current_index = 0
            self.update_display()
            self.update_pagination()
            messagebox.showinfo("Selesai", f"Selesai! {len(self.current_images)} gambar telah dikenali.")

    def update_display(self):
        if not self.current_images or self.current_index >= len(self.current_images):
            return
            
        img_path = self.current_images[self.current_index]
        res = self.results[self.current_index]
        
        # Tampilkan Gambar
        try:
            img = Image.open(img_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.lbl_image.config(image=photo)
            self.lbl_image.image = photo
        except Exception as e:
            self.lbl_image.config(text="Gagal memuat gambar")

        # Tampilkan Teks
        color = "#27ae60" if res["conf"] > 60 else "#e74c3c"
        teks = f"TERDETEKSI SEBAGAI: {res['nama']} (Confidence: {res['conf']:.1f}%)"
        self.lbl_result.config(text=teks, fg=color)
        
        self.lbl_filename.config(text=f"File: {os.path.basename(img_path)}\n(Jarak Manhattan: {res['jarak']:.4f})")

    def update_pagination(self):
        total = len(self.current_images)
        self.lbl_page.config(text=f"{self.current_index + 1} / {total}")
        
        self.btn_prev.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_index < total - 1 else tk.DISABLED)

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            self.update_pagination()

    def next_image(self):
        if self.current_index < len(self.current_images) - 1:
            self.current_index += 1
            self.update_display()
            self.update_pagination()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionUI(root)
    root.mainloop()
