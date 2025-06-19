import cv2
import numpy as np
import os

image_path = "picture/sudoku.jpg"
# === 1. 載入圖片 ===
image = cv2.imread(image_path)
if image is None:
    print("找不到圖片，請檢查檔名或路徑。")
    exit()

# === 2. 灰階處理 + 自適應二值化 ===
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               11, 10)

# === 3. 找最大輪廓 ===
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)
peri = cv2.arcLength(max_contour, True)
approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

if len(approx) != 4:
    print("❌ 沒找到正確的四邊形，請檢查圖片。")
    exit()
else:
    print("✅ 找到九宮格了！")

# === 4. 將點重新排序為：左上、右上、右下、左下 ===
def reorder_points(pts):
    pts = pts.reshape((4, 2))
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1)

    tl = pts[np.argmin(sum_pts)]
    br = pts[np.argmax(sum_pts)]
    tr = pts[np.argmin(diff_pts)]
    bl = pts[np.argmax(diff_pts)]

    return np.array([tl, tr, br, bl], dtype="float32")

# === 5. 做透視轉換 ===
def warp(img, points):
    pts1 = reorder_points(points)
    side = 450
    pts2 = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (side, side))
    return warped

warped = warp(gray, approx)  # 灰階即可
#cv2.imshow("矯正後九宮格", warped)
cv2.imwrite("warped_grid.png", warped)
print("✅ 已儲存矯正後九宮格圖片為 warped_grid.png")
cv2.waitKey(0)

# === 6. 切成 9x9 小格，去邊裁切 ===
cell_size = 450 // 9
padding = 4  # 🔧 去除邊框的像素數

os.makedirs("cells", exist_ok=True)
for i in range(9):
    for j in range(9):
        x = j * cell_size
        y = i * cell_size
        cell = warped[y:y+cell_size, x:x+cell_size]

        # ✅ 裁切掉四周 padding（避免線條干擾）
        cell = cell[padding:cell_size - padding, padding:cell_size - padding]

        filename = f"cells/cell_{i}_{j}.png"
        cv2.imwrite(filename, cell)

cv2.destroyAllWindows()
