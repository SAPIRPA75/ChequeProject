from PIL import Image
import pytesseract
import numpy as np
import cv2

filename = 'image_01.png'
img1 = np.array(Image.open(filename))
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(img1)

img = cv2.imread("image_noise.png")
cv2.imshow('Un- Normalized Image', img)

img = cv2.imread("image_noise.png")
norm = np.zeros((800,800))
final = cv2.normalize(img,  norm, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('Normalized Image', final)
cv2.imwrite('city_normalized.jpg', final)
cv2.waitKey(5000)



#import cv2norm_img = np.zeros((img1.shape[0], img1.shape[1]))
#img = cv2.normalize(img1, norm_img, 0, 255, cv2.NORM_MINMAX)
#img = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)[1]
#img = cv2.GaussianBlur(img, (1, 1), 0)


#print(text)