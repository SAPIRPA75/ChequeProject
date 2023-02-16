from PIL import Image
import pytesseract
import numpy as np


filename = 'image_01.png'
img1 = np.array(Image.open(filename))
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(img1)
print(text)