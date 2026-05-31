import numpy as np
from PIL import Image
import math
from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.face_utils import gaussian_blur_manual, sobel_edge_detection, skin_segmentation_manual

class FaceFeatureExtractor:
    def __init__(self, target_size=(100, 100), grid_size=4):
        self.target_size = target_size
        self.grid_size = grid_size

    def compute_lbp_fast(self, img):
        lbp = np.zeros_like(img, dtype=np.uint8)
        center = img[1:-1, 1:-1]
        lbp[1:-1, 1:-1] |= ((img[0:-2, 0:-2] >= center).astype(np.uint8) << 7)
        lbp[1:-1, 1:-1] |= ((img[0:-2, 1:-1] >= center).astype(np.uint8) << 6)
        lbp[1:-1, 1:-1] |= ((img[0:-2, 2:] >= center).astype(np.uint8) << 5)
        lbp[1:-1, 1:-1] |= ((img[1:-1, 2:] >= center).astype(np.uint8) << 4)
        lbp[1:-1, 1:-1] |= ((img[2:, 2:] >= center).astype(np.uint8) << 3)
        lbp[1:-1, 1:-1] |= ((img[2:, 1:-1] >= center).astype(np.uint8) << 2)
        lbp[1:-1, 1:-1] |= ((img[2:, 0:-2] >= center).astype(np.uint8) << 1)
        lbp[1:-1, 1:-1] |= ((img[1:-1, 0:-2] >= center).astype(np.uint8) << 0)
        return lbp

    def extract_features(self, face_rgb_100, eyes_rects_100):
        h, w, _ = face_rgb_100.shape
        
        # 1. PREPROCESSING
        gray = rgb_to_gray_manual(face_rgb_100)
        gray_eq = equalize_hist(gray)
        blurred = gaussian_blur_manual(gray_eq)
        
        # 2. SEGMENTASI KULIT (HSV Thresholding)
        pil_img = Image.fromarray(face_rgb_100)
        pil_hsv = np.array(pil_img.convert('HSV'))
        hsv = np.zeros_like(pil_hsv)
        hsv[:, :, 0] = (pil_hsv[:, :, 0].astype(np.float32) * 179 / 255).astype(np.uint8)
        hsv[:, :, 1] = pil_hsv[:, :, 1]
        hsv[:, :, 2] = pil_hsv[:, :, 2]
        skin_mask = skin_segmentation_manual(hsv)
        
        # 3. SEGMENTASI STRUKTUR WAJAH
        edges = sobel_edge_detection(blurred)
        face_contour = sobel_edge_detection(skin_mask)
        
        # --- A. Fitur Geometris (Rasio Baru) ---
        c1, c2 = None, None
        if len(eyes_rects_100) >= 2:
            eyes = sorted(eyes_rects_100[:2], key=lambda e: e[0])
            e1, e2 = eyes[0], eyes[1]
            c1 = (int(e1[0] + e1[2]//2), int(e1[1] + e1[3]//2))
            c2 = (int(e2[0] + e2[2]//2), int(e2[1] + e2[3]//2))
            eye_dist = math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
            eye_y_center = (c1[1] + c2[1]) / 2.0
            mid_x = (c1[0] + c2[0]) / 2.0
            
            # Dinamis berdasarkan jarak mata (eye_dist)
            nose_center = (int(mid_x), int(eye_y_center + 0.65 * eye_dist))
            mouth_center = (int(mid_x), int(eye_y_center + 1.15 * eye_dist))
        else:
            c1, c2 = (30, 40), (70, 40)
            eye_dist, eye_y_center = 40.0, 40.0
            nose_center = (50, 60)
            mouth_center = (50, 85)

        eye_to_nose_dist = abs(nose_center[1] - eye_y_center)
        nose_to_mouth_dist = abs(mouth_center[1] - nose_center[1])
        
        eye_to_nose_ratio = float(eye_to_nose_dist / (eye_dist + 1e-6))
        nose_to_mouth_ratio = float(nose_to_mouth_dist / (eye_dist + 1e-6))
        
        # --- B. Fitur Region Klasik ---
        eye_area_skin = skin_mask[0:int(eye_y_center+10), 0:100]
        eye_area_edge = edges[0:int(eye_y_center+10), 0:100]
        area_eye_size = 100 * max(1, int(eye_y_center+10))
        eye_non_skin_ratio = float(np.sum(eye_area_skin == 0)) / area_eye_size
        eye_edge_density = float(np.sum(eye_area_edge > 50)) / area_eye_size
        
        # Bounding box dinamis untuk hidung (ukuran 30x30)
        nx, ny = nose_center
        n_x1, n_y1 = max(0, nx - 15), max(0, ny - 15)
        n_x2, n_y2 = min(100, nx + 15), min(100, ny + 15)
        nose_region_edge = edges[n_y1:n_y2, n_x1:n_x2]
        nose_area_size = max(1, (n_x2 - n_x1) * (n_y2 - n_y1))
        nose_edge_density = float(np.sum(nose_region_edge > 50)) / nose_area_size
        nose_box = [n_x1, n_y1, n_x2, n_y2]
        
        # Bounding box dinamis untuk mulut (ukuran 50x25)
        mx, my = mouth_center
        m_x1, m_y1 = max(0, mx - 25), max(0, my - 10)
        m_x2, m_y2 = min(100, mx + 25), min(100, my + 15)
        mouth_region_skin = skin_mask[m_y1:m_y2, m_x1:m_x2]
        mouth_area_size = max(1, (m_x2 - m_x1) * (m_y2 - m_y1))
        mouth_non_skin_ratio = float(np.sum(mouth_region_skin == 0)) / mouth_area_size
        mouth_box = [m_x1, m_y1, m_x2, m_y2]
        
        # --- C. Grid Features (4x4) ---
        grid_edge_density = []
        grid_skin_ratio = []
        grid_intensity = []
        step_y, step_x = h // self.grid_size, w // self.grid_size
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                b_edge = edges[i*step_y:(i+1)*step_y, j*step_x:(j+1)*step_x]
                b_skin = skin_mask[i*step_y:(i+1)*step_y, j*step_x:(j+1)*step_x]
                b_gray = gray_eq[i*step_y:(i+1)*step_y, j*step_x:(j+1)*step_x]
                
                grid_edge_density.append(float(np.sum(b_edge > 50)) / (step_x*step_y))
                grid_skin_ratio.append(float(np.sum(b_skin > 0)) / (step_x*step_y))
                grid_intensity.append(float(np.mean(b_gray)))
                
        # --- D. Simetri Wajah ---
        L_gray = gray_eq[:, 0:w//2].astype(np.float32)
        R_gray = gray_eq[:, w//2:w].astype(np.float32)
        R_flipped = np.fliplr(R_gray)
        
        diff = np.abs(L_gray - R_flipped)
        sm = L_gray + R_flipped
        # Normalization factor avoids div by zero
        symmetry_error = float(np.sum(diff) / (np.sum(sm) + 1e-6))
        
        # --- E. LBP Histogram ---
        lbp = self.compute_lbp_fast(gray_eq)
        # Menggunakan 32 bin untuk histogram LBP agar tidak terlalu besar di JSON
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=32, range=[0, 256])
        lbp_hist = (lbp_hist / lbp_hist.sum()).tolist()
        
        return {
            "features": {
                "eye_distance": round(eye_dist / 100.0, 4),
                "eye_coords": [[c[0]/100.0, c[1]/100.0] for c in [c1, c2]],
                "nose_coords": [[nose_center[0]/100.0, nose_center[1]/100.0]],
                "nose_box": nose_box,
                "mouth_coords": [mouth_center],
                "mouth_box": mouth_box,
                "eye_to_nose_ratio": round(eye_to_nose_ratio, 4),
                "nose_to_mouth_ratio": round(nose_to_mouth_ratio, 4),
                
                "skin_ratio": round(float(np.sum(skin_mask > 0)) / (w*h), 4),
                "edge_density": round(float(np.sum(edges > 50)) / (w*h), 4),
                "contour_density": round(float(np.sum(face_contour > 50)) / (w*h), 4),
                
                "eye_region_edge": round(eye_edge_density, 4),
                "eye_non_skin": round(eye_non_skin_ratio, 4),
                "nose_region_edge": round(nose_edge_density, 4),
                "mouth_non_skin": round(mouth_non_skin_ratio, 4),
                
                "symmetry_error": round(symmetry_error, 4),
                
                "grid_edge_density": [round(x, 4) for x in grid_edge_density],
                "grid_skin_ratio": [round(x, 4) for x in grid_skin_ratio],
                "grid_intensity": [round(x, 4) for x in grid_intensity],
                
                "lbp_histogram": [round(x, 4) for x in lbp_hist]
            },
            "masks": {
                "skin": skin_mask,
                "edge": edges,
                "contour": face_contour,
                "lbp": lbp
            }
        }
