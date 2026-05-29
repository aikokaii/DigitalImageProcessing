import os
import json
from PIL import Image, ImageDraw, ImageFont

def buat_semua_visualisasi():
    db_path = "database_fitur.json"
    folder_wajah = "hasil_deteksi"
    folder_output = "hasil_visualisasi"

    os.makedirs(folder_output, exist_ok=True)

    if not os.path.exists(db_path):
        print(f"File {db_path} tidak ditemukan.")
        return

    with open(db_path, 'r') as f:
        db = json.load(f)

    if not os.path.exists(folder_wajah):
        print(f"Folder {folder_wajah} tidak ditemukan.")
        return

    all_files = [f for f in os.listdir(folder_wajah) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                 and '_skin' not in f.lower()
                 and '_edge' not in f.lower()]

    # Coba muat font Arial, jika tidak ada gunakan default bawaan PIL
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    print(f"Mulai membuat visualisasi untuk semua gambar...")

    count = 0
    for person_name, features_list in db.items():
        person_files = []
        for f in all_files:
            nm = os.path.splitext(f)[0]
            nm = nm.replace('_face detected', '').split('(')[0].strip().lower()
            if nm == person_name:
                person_files.append(f)
                
        person_files.sort()
        
        for i in range(min(len(person_files), len(features_list))):
            filename = person_files[i]
            file_path = os.path.join(folder_wajah, filename)
            features = features_list[i]
            
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize((100, 100))
                
                # Skalakan 4x lipat agar garis dan teks tidak pecah
                s = 4
                img = img.resize((100 * s, 100 * s), Image.NEAREST)
                draw = ImageDraw.Draw(img)
                
                # ==========================================
                # GAMBAR OVERLAY
                # ==========================================
                r = 4 # Jari-jari titik
                
                # 1. MATA
                c1, c2 = None, None
                if "eye_coords" in features and len(features["eye_coords"]) >= 2:
                    c1 = features["eye_coords"][0]
                    c2 = features["eye_coords"][1]
                    
                    # Garis antar mata
                    draw.line([(c1[0]*s, c1[1]*s), (c2[0]*s, c2[1]*s)], fill="cyan", width=2)
                    
                    # Titik mata
                    draw.ellipse([c1[0]*s-r, c1[1]*s-r, c1[0]*s+r, c1[1]*s+r], fill="cyan", outline="white")
                    draw.ellipse([c2[0]*s-r, c2[1]*s-r, c2[0]*s+r, c2[1]*s+r], fill="cyan", outline="white")

                # 2. HIDUNG
                nose = None
                if "nose_coords" in features and len(features["nose_coords"]) > 0:
                    nose = features["nose_coords"][0]
                    
                    # Kotak Area Hidung
                    if "nose_box" in features:
                        nx1, ny1, nx2, ny2 = features["nose_box"]
                        draw.rectangle([nx1*s, ny1*s, nx2*s, ny2*s], outline="lime", width=2)
                    
                    # Garis dari tengah mata ke hidung
                    if c1 and c2:
                        mid_x = (c1[0] + c2[0]) / 2 * s
                        mid_y = (c1[1] + c2[1]) / 2 * s
                        draw.line([(mid_x, mid_y), (nose[0]*s, nose[1]*s)], fill="yellow", width=2)
                        
                    # Titik hidung
                    draw.ellipse([nose[0]*s-r, nose[1]*s-r, nose[0]*s+r, nose[1]*s+r], fill="lime", outline="white")

                # 3. MULUT
                if "mouth_coords" in features and len(features["mouth_coords"]) > 0:
                    mouth = features["mouth_coords"][0]
                    
                    # Kotak Area Mulut
                    if "mouth_box" in features:
                        mx1, my1, mx2, my2 = features["mouth_box"]
                        draw.rectangle([mx1*s, my1*s, mx2*s, my2*s], outline="red", width=2)
                        
                    # Garis dari hidung ke mulut
                    if nose:
                        draw.line([(nose[0]*s, nose[1]*s), (mouth[0]*s, mouth[1]*s)], fill="magenta", width=2)
                        
                    # Titik mulut
                    draw.ellipse([mouth[0]*s-r, mouth[1]*s-r, mouth[0]*s+r, mouth[1]*s+r], fill="red", outline="white")
                
                # 4. TEKS INFORMASI (Pojok Kiri Atas)
                texts = []
                if "eye_distance" in features: texts.append(f"Jarak Mata: {features['eye_distance']}")
                if "skin_ratio" in features: texts.append(f"Rasio Kulit: {features['skin_ratio']}")
                if "symmetry_error" in features: texts.append(f"Simetri: {features['symmetry_error']}")
                
                y_offset = 10
                for txt in texts:
                    # Gambar bayangan teks agar selalu terbaca
                    draw.text((11, y_offset+1), txt, fill="black", font=font)
                    draw.text((10, y_offset), txt, fill="white", font=font)
                    y_offset += 20
                    
                # SIMPAN GAMBAR
                out_path = os.path.join(folder_output, filename)
                img.save(out_path)
                count += 1
                
            except Exception as e:
                print(f"Gagal memproses {filename}: {e}")

    print(f"\nSelesai! Berhasil menyimpan {count} gambar visualisasi ke dalam folder '{folder_output}'.")

if __name__ == "__main__":
    buat_semua_visualisasi()
