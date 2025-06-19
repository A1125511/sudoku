#該程式碼無法識別
import cv2
import easyocr 

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 去除小雜點與線
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return clean

reader = easyocr.Reader(['en'])
board = [[0 for _ in range(9)] for _ in range(9)]

for row in range(9):
    for col in range(9):
        img_path = f'cells/cell_{row}_{col}.png'
        img = preprocess_image(img_path)  # ⚠ 使用處理過的圖片
        result = reader.readtext(img, detail=1)
        
        if result:
            text = result[0][1]  # ⚠ 取出辨識的文字
            try:
                number = int(text)
                board[row][col] = number
            except ValueError:
                pass  # 非數字則略過

# 印出結果
for row in board:
    print(row)
