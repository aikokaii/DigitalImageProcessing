import os
import json
import tkinter as tk
from PIL import Image, ImageTk

class FeatureVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Visualisasi Overlay Fitur Wajah")
        
        self.db_path = "database_fitur.json"
        self.folder_wajah = "hasil_deteksi"
        
        # Array untuk menyimpan data gabungan (gambar + fitur)
        self.data_map = [] 
        self.current_idx = 0
        
        # Perbesar ukuran gambar saat ditampilkan agar lebih jelas (aslinya 100x100)
        self.scale_factor = 4
        self.img_size = 100 * self.scale_factor
        
        self._load_data()
        
        # --- Bikin UI (User Interface) ---
        self.canvas = tk.Canvas(master, width=self.img_size, height=self.img_size, bg="black")
        self.canvas.pack(pady=10, padx=10)
        
        self.info_label = tk.Label(master, text="Memuat...", font=("Arial", 12, "bold"))
        self.info_label.pack(pady=5)
        
        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=10)
        
        self.btn_prev = tk.Button(btn_frame, text="<< Prev", command=self.prev_img, width=15, font=("Arial", 10))
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        
        self.btn_next = tk.Button(btn_frame, text="Next >>", command=self.next_img, width=15, font=("Arial", 10))
        self.btn_next.pack(side=tk.LEFT, padx=10)
        
        # Info legenda
        legend_text = "Biru: Mata | Hijau: Hidung | Merah: Mulut\nGaris Kuning/Pink: Jarak Relatif Geometri"
        self.legend_label = tk.Label(master, text=legend_text, fg="gray", font=("Arial", 10))
        self.legend_label.pack(pady=5)
        
        # Tampilkan gambar pertama jika data ada
        if self.data_map:
            self.show_image()
        else:
            self.info_label.config(text="Data JSON atau Gambar tidak ditemukan!")
            self.btn_next.config(state=tk.DISABLED)
            self.btn_prev.config(state=tk.DISABLED)
            
    def _load_data(self):
        """Membaca JSON dan mencocokkannya dengan file gambar di hasil_deteksi."""
        if not os.path.exists(self.db_path):
            print(f"File {self.db_path} tidak ditemukan.")
            return
            
        with open(self.db_path, 'r') as f:
            try:
                db = json.load(f)
            except Exception as e:
                print(f"Gagal membaca JSON: {e}")
                return
            
        if not os.path.exists(self.folder_wajah):
            return
            
        all_files = [f for f in os.listdir(self.folder_wajah) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                     and '_skin' not in f.lower()
                     and '_edge' not in f.lower()]
        
        # Hubungkan fitur di database dengan file gambar aslinya
        for person_name, features_list in db.items():
            person_files = []
            for f in all_files:
                # Ambil nama bersih untuk pencocokan
                nm = os.path.splitext(f)[0]
                nm = nm.replace('_face detected', '').split('(')[0].strip().lower()
                
                if nm == person_name:
                    person_files.append(f)
                    
            # Sort berdasar abjad, sesuai urutan saat disimpan di ekstraksi_fitur.py
            person_files.sort()
            
            # Gabungkan file dan fitur yang bersesuaian
            for i in range(min(len(person_files), len(features_list))):
                self.data_map.append({
                    'file': os.path.join(self.folder_wajah, person_files[i]),
                    'filename': person_files[i],
                    'name': person_name,
                    'features': features_list[i]
                })
                
    def show_image(self):
        """Menampilkan gambar saat ini dan menggambar overlay titik/garis."""
        self.canvas.delete("all") # Bersihkan kanvas
        
        if not self.data_map:
            return
            
        item = self.data_map[self.current_idx]
        features = item['features']
        
        # Update teks info
        self.info_label.config(text=f"[{self.current_idx+1}/{len(self.data_map)}] {item['filename']} (Subjek: {item['name'].upper()})")
        
        # Update state tombol
        self.btn_prev.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_idx < len(self.data_map)-1 else tk.DISABLED)
        
        # 1. Load Image
        try:
            pil_img = Image.open(item['file']).convert('RGB')
            # Gambar ekstraksi fitur menggunakan base 100x100
            pil_img = pil_img.resize((100, 100))
            # Perbesar untuk visualisasi di layar komputer
            pil_img = pil_img.resize((self.img_size, self.img_size), Image.NEAREST)
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        except Exception as e:
            print(f"Gagal memuat gambar {item['file']}: {e}")
            return
            
        # 2. Gambar Overlay Koordinat Fitur
        s = self.scale_factor # Pengali koordinat asli 100x100 ke ukuran display
        
        # Fitur Koordinat Mata
        c1, c2 = None, None
        if "eye_coords" in features and len(features["eye_coords"]) >= 2:
            c1 = features["eye_coords"][0]
            c2 = features["eye_coords"][1]
            
            self._draw_point(c1[0]*s, c1[1]*s, color="#00FFFF") # Kiri
            self._draw_point(c2[0]*s, c2[1]*s, color="#00FFFF") # Kanan
            
            # Garis Jarak Mata
            self.canvas.create_line(c1[0]*s, c1[1]*s, c2[0]*s, c2[1]*s, fill="#00FFFF", width=2, dash=(4,2))
            
        # Fitur Koordinat Hidung
        nose = None
        if "nose_coords" in features and len(features["nose_coords"]) > 0:
            nose = features["nose_coords"][0]
            self._draw_point(nose[0]*s, nose[1]*s, color="#00FF00") # Hijau
            
            # Kotak hidung (jika ada)
            if "nose_box" in features:
                nx1, ny1, nx2, ny2 = features["nose_box"]
                self.canvas.create_rectangle(nx1*s, ny1*s, nx2*s, ny2*s, outline="#00FF00", width=1, dash=(2,2))
            
            # Garis Segitiga dari titik tengah mata ke Hidung
            if c1 and c2:
                mid_eye_x = (c1[0] + c2[0]) / 2 * s
                mid_eye_y = (c1[1] + c2[1]) / 2 * s
                self.canvas.create_line(mid_eye_x, mid_eye_y, nose[0]*s, nose[1]*s, fill="yellow", width=2, dash=(4,2))
                
        # Fitur Koordinat Mulut
        mouth = None
        if "mouth_coords" in features and len(features["mouth_coords"]) > 0:
            mouth = features["mouth_coords"][0]
            self._draw_point(mouth[0]*s, mouth[1]*s, color="#FF0000") # Merah
            
            # Kotak mulut (jika ada)
            if "mouth_box" in features:
                mx1, my1, mx2, my2 = features["mouth_box"]
                self.canvas.create_rectangle(mx1*s, my1*s, mx2*s, my2*s, outline="#FF0000", width=1, dash=(2,2))
            
            # Garis dari hidung ke mulut
            if nose:
                self.canvas.create_line(nose[0]*s, nose[1]*s, mouth[0]*s, mouth[1]*s, fill="#FF00FF", width=2, dash=(4,2))
                
        # 3. Tampilkan Teks Metadata Tambahan di Pojok Kiri Atas
        texts = []
        if "eye_distance" in features: texts.append(f"Jarak Mata: {features['eye_distance']}")
        if "skin_ratio" in features: texts.append(f"Rasio Kulit: {features['skin_ratio']}")
        if "symmetry_error" in features: texts.append(f"Simetri: {features['symmetry_error']}")
        
        y_offset = 10
        for txt in texts:
            # Beri efek bayangan (shadow) agar teks terbaca di background gelap/terang
            self.canvas.create_text(11, y_offset+1, anchor=tk.NW, text=txt, fill="black", font=("Consolas", 10, "bold"))
            self.canvas.create_text(10, y_offset, anchor=tk.NW, text=txt, fill="white", font=("Consolas", 10, "bold"))
            y_offset += 15

    def _draw_point(self, x, y, color="red", r=5):
        """Fungsi pembantu untuk menggambar titik bulat (lingkaran)."""
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline="white", width=1.5)

    def next_img(self):
        if self.current_idx < len(self.data_map) - 1:
            self.current_idx += 1
            self.show_image()

    def prev_img(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_image()

if __name__ == "__main__":
    # Pakai Tkinter agar lebih mudah dijalankan di Windows tanpa install lib aneh-aneh
    root = tk.Tk()
    
    # Agar jendela muncul di tengah layar
    window_width, window_height = 500, 600
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    x_cordinate = int((screen_width/2) - (window_width/2))
    y_cordinate = int((screen_height/2) - (window_height/2))
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
    
    app = FeatureVisualizer(root)
    root.mainloop()
