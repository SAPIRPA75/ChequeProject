from PIL import Image
import pytesseract
import numpy as np
import cv2



def Extract_string_From_Image(filename):
    img = np.array(Image.open(filename))
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(img)
    return text

filename = 'image_01.png'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sapir\PycharmProjects\cheque_detection\Tesseract-OCR\tesseract.exe'
image = cv2.imread(filename)
cv2.imshow("The Image is:",image)
cv2.waitKey(0)
results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

for i in range(0, len(results["text"])):
   x = results["left"][i]
   y = results["top"][i]

   w = results["width"][i]
   h = results["height"][i]

   text = results["text"][i]
   conf = int(results["conf"][i])

   if conf > 70:
       text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
       cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
       cv2.putText(image, text, (x, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

cv2.imshow("The Boundries:",image)
cv2.waitKey(0)

#Proccessed_img = Extract_string_From_Image('image_01.png')
#print(Proccessed_img)


#img = cv2.imread("image_noise.png")
#cv2.imshow('Un- Normalized Image', img)
#cv2.waitKey(0)
#cv2norm_img = np.zeros((img.shape[0], img.shape[1]))
#norm = np.zeros((800,800))
#final = cv2.normalize(img,  norm, 0, 255, cv2.NORM_MINMAX)
#cv2.imshow('Normalized Image', final)
#if (not(cv2.imwrite('city_normalized.jpg', final))):
#    print("Image didnt saved")
#img = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)[1]
#img = cv2.GaussianBlur(img, (1, 1), 0)
#cv2.imshow('Un- Normalized Image', img)
#cv2.waitKey(0)




#import cv2norm_img = np.zeros((img1.shape[0], img1.shape[1]))
#img = cv2.normalize(img1, norm_img, 0, 255, cv2.NORM_MINMAX)
#img = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)[1]
#img = cv2.GaussianBlur(img, (1, 1), 0)


#print(text)