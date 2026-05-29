import numpy as np
import xml.etree.ElementTree as ET

def compute_integral_image(img):
    """
    Hitung integral image dan squared integral image.
    Input : img  (2D numpy array, dtype float64)
    Output: (ii, ii_sq) masing-masing shape (H+1, W+1)
    """
    h, w = img.shape
    # Prefix sum dengan baris/kolom nol di depan (standar integral image)
    ii    = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii_sq = np.zeros((h + 1, w + 1), dtype=np.float64)

    ii[1:, 1:]    = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    ii_sq[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float64) ** 2, axis=0), axis=1)

    return ii, ii_sq


def rect_sum(ii, x, y, w, h):
    """Ambil jumlah piksel dalam persegi (x,y,w,h) dari integral image."""
    return ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]


# ============================================================
# HELPER: Resize gambar (tanpa cv2.resize) — bilinear interpolation
# ============================================================

def resize_bilinear(img, new_w, new_h):
    """
    Resize 2D grayscale image ke (new_w, new_h) menggunakan
    bilinear interpolation murni NumPy.
    """
    src_h, src_w = img.shape
    if src_w == new_w and src_h == new_h:
        return img.copy()

    # Koordinat target dalam skala sumber
    x_ratio = src_w / new_w
    y_ratio = src_h / new_h

    x = np.arange(new_w) * x_ratio
    y = np.arange(new_h) * y_ratio

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x0 = np.clip(x0, 0, src_w - 1)
    y0 = np.clip(y0, 0, src_h - 1)

    dx = (x - np.floor(x))[:, np.newaxis]   # (new_w, 1)
    dy = (y - np.floor(y))[np.newaxis, :]   # (1, new_h)

    # Interpolasi bilinear: axes → (new_w, new_h), lalu transpose ke (new_h, new_w)
    Ia = img[np.ix_(y0, x0)]   # top-left      (new_h, new_w)
    Ib = img[np.ix_(y1, x0)]   # bottom-left
    Ic = img[np.ix_(y0, x1)]   # top-right
    Id = img[np.ix_(y1, x1)]   # bottom-right

    dx_h = dx.T                  # (1, new_w) → broadcast over new_h
    dy_v = dy                    # (1, new_h) → will broadcast

    # Bilinear formula
    result = (
        Ia * (1 - dx_h) * (1 - dy_v.T) +
        Ic * dx_h       * (1 - dy_v.T) +
        Ib * (1 - dx_h) *  dy_v.T +
        Id * dx_h       *  dy_v.T
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# HELPER: Non-Maximum Suppression / Group Rectangles (tanpa cv2)
# ============================================================

def iou(r1, r2):
    """Hitung Intersection over Union dua rectangle [x,y,w,h]."""
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[0] + r1[2], r2[0] + r2[2])
    y2 = min(r1[1] + r1[3], r2[1] + r2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = r1[2] * r1[3]
    area2 = r2[2] * r2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def group_rectangles(rects, min_neighbors=3, iou_thresh=0.2):
    """
    Kelompokkan kotak-kotak yang tumpang-tindih, kembalikan
    hanya kotak yang didukung oleh >= min_neighbors tetangga.
    Pengganti cv2.groupRectangles — murni Python/NumPy.
    """
    if len(rects) == 0:
        return []

    n = len(rects)
    labels = [-1] * n

    # Union-Find sederhana
    def find(i):
        while labels[i] != i:
            labels[i] = labels[labels[i]]
            i = labels[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            labels[ri] = rj

    for i in range(n):
        labels[i] = i

    for i in range(n):
        for j in range(i + 1, n):
            if iou(rects[i], rects[j]) > iou_thresh:
                union(i, j)

    # Kumpulkan per kelompok
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(rects[i])

    result = []
    for members in groups.values():
        if len(members) >= min_neighbors:
            xs = [r[0] for r in members]
            ys = [r[1] for r in members]
            ws = [r[2] for r in members]
            hs = [r[3] for r in members]
            result.append([
                int(np.mean(xs)),
                int(np.mean(ys)),
                int(np.mean(ws)),
                int(np.mean(hs))
            ])

    return result


# ============================================================
# KELAS UTAMA: ManualCascadeClassifier
# ============================================================

class ManualCascadeClassifier:
    """
    Membaca file XML Haar Cascade (format OpenCV) dan menjalankan
    deteksi objek secara MURNI MANUAL — tanpa cv2 sama sekali.
    """

    def __init__(self, xml_path):
        self.stages = []
        self.features = []
        self.base_window_size = (24, 24)
        self.load_from_xml(xml_path)

    # ----------------------------------------------------------
    # Load XML
    # ----------------------------------------------------------

    def load_from_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        cascade = root.find('cascade')
        if cascade is None:
            raise ValueError("Invalid cascade XML: tidak ada node <cascade>")

        width  = int(cascade.find('width').text)
        height = int(cascade.find('height').text)
        self.base_window_size = (width, height)

        # Load fitur Haar
        features_node = cascade.find('features')
        for feat_node in features_node.findall('_'):
            rects = []
            for rect_node in feat_node.find('rects').findall('_'):
                parts = rect_node.text.strip().split()
                rects.append((
                    int(parts[0]),   # x
                    int(parts[1]),   # y
                    int(parts[2]),   # w
                    int(parts[3]),   # h
                    float(parts[4])  # weight
                ))
            self.features.append(rects)

        # Load stages
        stages_node = cascade.find('stages')
        for stage_node in stages_node.findall('_'):
            stage_thresh = float(stage_node.find('stageThreshold').text)
            weak_classifiers = []

            for weak_node in stage_node.find('weakClassifiers').findall('_'):
                internal = weak_node.find('internalNodes').text.strip().split()
                leaves   = weak_node.find('leafValues').text.strip().split()

                weak_classifiers.append({
                    'feature_idx': int(internal[2]),
                    'threshold':   float(internal[3]),
                    'leaf0':       float(leaves[0]),
                    'leaf1':       float(leaves[1])
                })

            self.stages.append({
                'threshold':        stage_thresh,
                'weak_classifiers': weak_classifiers
            })

    # ----------------------------------------------------------
    # Deteksi Utama
    # ----------------------------------------------------------

    def detectMultiScale(self, image, scaleFactor=1.2, minNeighbors=3,
                         minSize=(30, 30)):
        """
        Deteksi objek pada `image` menggunakan sliding window multi-skala.
        image    : numpy 2D (grayscale)
        Kembalikan: list of [x, y, w, h]
        """
        # Pastikan grayscale
        if len(image.shape) == 3:
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        img_h, img_w = gray.shape
        win_w, win_h = self.base_window_size
        inv_area     = 1.0 / (win_w * win_h)

        detected = []

        scale = 1.0
        while True:
            scaled_w = int(img_w / scale)
            scaled_h = int(img_h / scale)

            if scaled_w < max(minSize[0], win_w) or scaled_h < max(minSize[1], win_h):
                break

            # Resize manual (tanpa cv2)
            scaled_img = resize_bilinear(gray, scaled_w, scaled_h).astype(np.float64)

            # Integral image manual (tanpa cv2.integral2)
            ii, ii_sq = compute_integral_image(scaled_img)

            step = max(1, win_w // 8)   # langkah sliding window

            for y in range(0, scaled_h - win_h, step):
                for x in range(0, scaled_w - win_w, step):

                    # Hitung mean & std pada jendela (untuk normalisasi)
                    win_sum    = rect_sum(ii,    x, y, win_w, win_h)
                    win_sq_sum = rect_sum(ii_sq, x, y, win_w, win_h)

                    mean     = win_sum * inv_area
                    variance = win_sq_sum * inv_area - mean * mean
                    std      = float(np.sqrt(max(0.0, variance))) or 1.0

                    # Evaluasi setiap stage cascade
                    pass_all = True
                    for stage in self.stages:
                        stage_sum = 0.0
                        for weak in stage['weak_classifiers']:
                            feature   = self.features[weak['feature_idx']]
                            feat_val  = 0.0

                            for rx, ry, rw, rh, rweight in feature:
                                # Pastikan tidak keluar batas integral image
                                if (x + rx + rw) <= scaled_w and (y + ry + rh) <= scaled_h:
                                    feat_val += rect_sum(ii, x + rx, y + ry, rw, rh) * rweight

                            feat_val *= inv_area

                            if feat_val < weak['threshold'] * std:
                                stage_sum += weak['leaf0']
                            else:
                                stage_sum += weak['leaf1']

                        if stage_sum < stage['threshold']:
                            pass_all = False
                            break

                    if pass_all:
                        orig_x = int(x * scale)
                        orig_y = int(y * scale)
                        orig_w = int(win_w * scale)
                        orig_h = int(win_h * scale)
                        detected.append([orig_x, orig_y, orig_w, orig_h])

            scale *= scaleFactor

        # Group rectangles (tanpa cv2.groupRectangles)
        return group_rectangles(detected, min_neighbors=minNeighbors, iou_thresh=0.2)