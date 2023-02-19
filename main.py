from PIL import Image
import pytesseract
import numpy as np
import cv2
from matplotlib import pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sapir\PycharmProjects\cheque_detection\Tesseract-OCR\tesseract.exe'


def Extract_string_From_Image(Proccessed_img):
    text = pytesseract.image_to_string(Proccessed_img)
    return text

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def Remove_Noise(image):
    return cv2.medianBlur(image,5)

def thresholding(image):
    return cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

'''''
image = cv2.imread('big_image.jpg')
template = cv2.imread('template.jpg')

image_gray = cv2.rgb2gray(image)
template_gray = cv2.rgb2gray(template)

result = cv2.match_template(image_gray, template_gray)
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]
'''''


'''''
img_rgb = cv2.imread('cheque_Example.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
cv2.imshow("BGR2GREY", img_gray)
cv2.waitKey(0)
template = cv2.imread('tamplate-2.png')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cv2.imshow("BGR2GREY", template_gray)
cv2.waitKey(0)
w, h = template_gray.shape[::-1]
res = cv2.matchTemplate(img_gray,template_gray,cv2.TM_CCOEFF_NORMED)
threshold = 0.2
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv2.imwrite('res.png',img_rgb)
cv2.imshow("The final is:",img_rgb)
cv2.waitKey(0)

'''''
img = cv2.imread('Mizrahi_bank_cheque.jpg',0)
img2 = img.copy()
template = cv2.imread('Mizrahi_bank_cheque_template.png',0)
w, h = template.shape[::-1]

# Apply template Matching = 'cv2.TM_SQDIFF' = '0'
res = cv2.matchTemplate(img,template,0)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

 # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
original_img = cv2.imread('Mizrahi_bank_cheque.jpg')
result = cv2.rectangle(original_img,top_left, bottom_right, 255, 2)
cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''''
 plt.subplot(121),plt.imshow(res,cmap = 'gray')
 plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
 plt.subplot(122),plt.imshow(img,cmap = 'gray')
 plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(meth)
plt.show()
'''''

'''''
image = cv2.imread('cheque_Example.jpg ')
Proccessed_img = get_grayscale(image)
Proccessed_img = thresholding(Proccessed_img)
Proccessed_img = Remove_Noise(Proccessed_img)
text_From_Image = Extract_string_From_Image(Proccessed_img)
print(text_From_Image)

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
'''