import numpy as np
import os
import sys
from PIL import Image
from src.face_utils import gaussian_blur_manual, sobel_edge_detection, skin_segmentation_manual
from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import equalize_hist, rgb_to_gray_manual

# =========================
# LOAD CASCADE
# =========================
cv2_data_dir = r"C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\cv2\data"
if not os.path.exists(cv2_data_dir):
    # Fallback to site-packages path
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        potential_path = os.path.join(sp, 'cv2', 'data')
        if os.path.exists(potential_path):
            cv2_data_dir = potential_path
            break

print("Loading manual Haar Cascade for Face from XML... (this may take a moment)")
face_cascade = ManualCascadeClassifier(os.path.join(cv2_data_dir, 'haarcascade_frontalface_default.xml'))
print("Loading manual Haar Cascade for Eyes from XML...")
eye_cascade = ManualCascadeClassifier(os.path.join(cv2_data_dir, 'haarcascade_eye.xml'))

# =========================
# FOLDER INPUT / OUTPUT
# =========================
input_folder = "webcam"
output_skin = "hasil_skin"
output_edge = "hasil_edge"
output_detection = "hasil_deteksi"

os.makedirs(output_skin, exist_ok=True)
os.makedirs(output_edge, exist_ok=True)
os.makedirs(output_detection, exist_ok=True)

# =========================
# AMBIL SEMUA GAMBAR
# =========================
image_files = [f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".pgm")]

# =========================
# PROSES GAMBAR
# =========================
for image_name in image_files:
    print(f"Memproses: {image_name}")
    image_path = os.path.join(input_folder, image_name)
    try:
        pil_img = Image.open(image_path).convert('RGB')
        pil_img = pil_img.resize((320, 240))
        frame = np.array(pil_img) # Bekerja dalam RGB sekarang
    except Exception as e:
        print(f"Gagal memuat {image_name}: {e}")
        continue

    gray = rgb_to_gray_manual(frame)
    gray = equalize_hist(gray)
    
    print(f"-> Detecting faces using from-scratch cascade on {image_name}...")
    # NOTE: It is very slow in pure python!
    raw_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    # Filter wajah secara manual berdasarkan ukuran asli bounding box 
    # (karena minSize di manual cascade memiliki bug cara kerjanya)
    faces = [(x, y, w, h) for (x, y, w, h) in raw_faces if w >= 60 and h >= 60]

    if len(faces) == 0:
        print(f"Wajah tidak terdeteksi pada {image_name}, menggunakan fallback (seluruh gambar).")
        # Jika tidak terdeteksi, kita anggap seluruh gambar adalah wajah
        faces = [(0, 0, frame.shape[1], frame.shape[0])]
        
    final_crop = frame # Default ke full frame jika tidak ada wajah
    skin_mask = None
    edges = None
    
    for (x, y, w, h) in faces:
        # Perlebar kotak secara vertikal agar mulut tidak terpotong
        h_new = int(h * 1.25)
        # Pastikan tidak melebihi batas bawah gambar
        if y + h_new > frame.shape[0]:
            h_new = frame.shape[0] - y
            
        roi_gray = gray[y:y+h_new, x:x+w]
        roi_color = frame[y:y+h_new, x:x+w]
        
        # Crop bagian wajah saja!
        final_crop = roi_color 
        
        # Ekstrak skin dan edge dari hasil crop (agar bebas dari noise background)
        blurred_crop = gaussian_blur_manual(roi_gray)
        pil_img_crop = Image.fromarray(roi_color)
        pil_hsv_crop = np.array(pil_img_crop.convert('HSV'))
        hsv_crop = np.zeros_like(pil_hsv_crop)
        hsv_crop[:, :, 0] = (pil_hsv_crop[:, :, 0].astype(np.float32) * 179 / 255).astype(np.uint8)
        hsv_crop[:, :, 1] = pil_hsv_crop[:, :, 1]
        hsv_crop[:, :, 2] = pil_hsv_crop[:, :, 2]
        
        skin_mask = skin_segmentation_manual(hsv_crop)
        edges = sobel_edge_detection(blurred_crop)
        
        # NOTE: Eye detection perlu jalan di h asli atau h_new bebas
        h = h_new 

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=9, minSize=(20, 20))
        filtered_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey < h * 0.5 and ew > 15 and eh > 15]
        
        filtered_eyes = sorted(filtered_eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        filtered_eyes = sorted(filtered_eyes, key=lambda e: e[0])
        
        # Lingkaran tidak lagi digambar, tetapi variabel filtered_eyes tetap menyimpan koordinatnya!

    # Pisahkan dataset latih dan testing secara total
    if "datatest" in image_name.lower():
        target_folder = "datatest"
        out_skin = "datatest_skin"
        out_edge = "datatest_edge"
    else:
        target_folder = output_detection
        out_skin = output_skin
        out_edge = output_edge

    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(out_skin, exist_ok=True)
    os.makedirs(out_edge, exist_ok=True)

    filename = os.path.splitext(image_name)[0]

    if skin_mask is not None and edges is not None:
        Image.fromarray(skin_mask, mode='L').save(os.path.join(out_skin, filename + "_skin.png"))
        Image.fromarray(edges, mode='L').save(os.path.join(out_edge, filename + "_edge.png"))
        
    Image.fromarray(final_crop, mode='RGB').save(os.path.join(target_folder, f"{filename}_face detected.png"))
    print(f"Selesai: {image_name} (Tersimpan di folder '{target_folder}')")

print("Semua gambar berhasil diproses")