import os
import numpy as np
from PIL import Image

class EigenfaceRecognizer:
    def __init__(self, target_size=(100, 100), num_components=None):
        self.target_size = target_size
        self.num_components = num_components
        
        self.mean_face = None
        self.eigenfaces = None
        self.weights = None
        self.labels = []
        
    def _read_images(self, folder):
        """Membaca semua gambar hasil deteksi dan me-return matrix data serta label."""
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not files:
            print(f"Tidak ada gambar wajah di folder '{folder}'.")
            return None, None
            
        faces_matrix = []
        labels = []
        
        for f in files:
            # Hapus ekstensi, dan hapus angka dalam kurung, contoh: "rafi (1).png" -> "rafi"
            # atau "aldi_face detected.png" -> "aldi_face detected" -> "aldi"
            nama_mentah = os.path.splitext(f)[0]
            if '_face detected' in nama_mentah:
                nama_mentah = nama_mentah.replace('_face detected', '')
                
            label = nama_mentah.split('(')[0].strip().lower()
            path = os.path.join(folder, f)
            
            try:
                # Eigenface bekerja paling baik di grayscale
                pil_img = Image.open(path).convert('L')
                pil_img = pil_img.resize(self.target_size)
                
                # Ubah ke 1D array
                face_vec = np.array(pil_img).flatten()
                faces_matrix.append(face_vec)
                labels.append(label)
            except Exception as e:
                print(f"Gagal memuat {f}: {e}")
                
        # Konversi ke numpy array (M x D) dimana M=jumlah gambar, D=jumlah piksel
        faces_matrix = np.array(faces_matrix, dtype=np.float64)
        return faces_matrix, labels

    def train(self, folder_path):
        """Melatih model Eigenface menggunakan PCA dari gambar-gambar wajah."""
        print("Membaca data pelatihan...")
        X, self.labels = self._read_images(folder_path)
        
        if X is None or len(X) == 0:
            return False
            
        M, D = X.shape
        print(f"Total gambar: {M}, Resolusi/Dimensi: {D} piksel.")
        
        # 1. Hitung Mean Face (Wajah rata-rata)
        self.mean_face = np.mean(X, axis=0)
        
        # 2. Kurangi wajah dengan mean face (Center the data)
        A = X - self.mean_face
        
        # 3. Covariance matrix alternatif (A * A^T) agar lebih ringan
        # Ukuran asli A^T * A adalah (10000x10000), terlalu besar. 
        # A * A^T ukurannya hanya (M x M)
        L = np.dot(A, A.T)
        
        # 4. Hitung Eigenvalues dan Eigenvectors dari L
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Urutkan dari terbesar ke terkecil
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Batasi komponen utama (PCA) jika ditentukan, jika tidak gunakan semuanya
        if self.num_components is None or self.num_components > M:
            k = M
        else:
            k = self.num_components
            
        eigenvectors = eigenvectors[:, :k]
        
        # 5. Kembalikan ke ruang dimensi piksel (Eigenfaces sesungguhnya)
        # Rumus: U = A^T * V
        self.eigenfaces = np.dot(A.T, eigenvectors)
        
        # 6. Normalisasi eigenfaces
        for i in range(self.eigenfaces.shape[1]):
            self.eigenfaces[:, i] /= np.linalg.norm(self.eigenfaces[:, i])
            
        # 7. Hitung bobot tiap gambar latih (Proyeksi ke ruang eigenface)
        # weights = A * U
        self.weights = np.dot(A, self.eigenfaces)
        print("Pelatihan Eigenface selesai!\n")
        return True
        
    def predict(self, image_path, threshold=None):
        """Memprediksi wajah baru dengan memproyeksikannya ke ruang Eigenface."""
        try:
            pil_img = Image.open(image_path).convert('L')
            pil_img = pil_img.resize(self.target_size)
            test_vec = np.array(pil_img, dtype=np.float64).flatten()
        except Exception as e:
            print(f"Gagal memuat gambar tes: {e}")
            return None
            
        # 1. Kurangi dengan mean face
        test_centered = test_vec - self.mean_face
        
        # 2. Proyeksikan ke ruang eigenface
        test_weight = np.dot(test_centered, self.eigenfaces)
        
        # 3. Cari jarak Euclidean terdekat (L2 Norm)
        distances = []
        for i, train_weight in enumerate(self.weights):
            dist = np.linalg.norm(train_weight - test_weight)
            distances.append((dist, self.labels[i]))
            
        distances.sort(key=lambda x: x[0])
        best_dist, best_label = distances[0]
        
        dikenali = True
        if threshold is not None and best_dist > threshold:
            dikenali = False
            
        return {
            "dikenali": dikenali,
            "nama": best_label if dikenali else "TIDAK DIKENALI",
            "kandidat_terdekat": best_label,
            "jarak": best_dist,
            "semua_jarak": distances
        }

if __name__ == "__main__":
    # ==========================
    # DEMO PENGGUNAAN EIGENFACE
    # ==========================
    folder_wajah = "hasil_deteksi"
    
    if not os.path.exists(folder_wajah):
        print(f"Folder '{folder_wajah}' tidak ditemukan. Harap jalankan deteksi_wajah.py terlebih dahulu.")
        exit()
        
    print("="*40)
    print("     EIGENFACE RECOGNITION DEMO")
    print("="*40)
    
    # 1. Inisiasi dan latih
    ef = EigenfaceRecognizer()
    sukses = ef.train(folder_wajah)
    
    if sukses:
        folder_test = "datatest"
        if not os.path.exists(folder_test) or not os.listdir(folder_test):
            print(f"\nFolder '{folder_test}' tidak ditemukan atau kosong.")
        else:
            print(f"--- Tes Prediksi (Gambar dari folder {folder_test}) ---")
            # Ambil file pertama di folder datatest
            test_file = os.listdir(folder_test)[0]
            test_path = os.path.join(folder_test, test_file)
            
            hasil = ef.predict(test_path, threshold=5000) # Threshold bisa disesuaikan
            
            print(f"File Uji    : {test_file}")
            print(f"Prediksi    : {hasil['nama'].upper()}")
            print(f"Jarak (Euclidean): {hasil['jarak']:.4f}")
            
            print("\nUrutan Kandidat Terdekat:")
            for dist, label in hasil['semua_jarak'][:5]: # Tampilkan top 5
                print(f" -> {label:<15} (Jarak: {dist:.4f})")
