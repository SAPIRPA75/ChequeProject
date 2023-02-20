from PIL import Image
import pytesseract
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import PySimpleGUI as sg
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\sapirpa\PycharmProjects\tesseract.exe'

def Start_GUI():
    layout = [[sg.Text("Welcome To Cheque Identifier app , Upload your Cheque bank :\n")], [sg.Button("Leumi Bank")],
              [sg.Button("BinLeumi Bank")], [sg.Button("Discount Bank")], [sg.Button("Mizrahi Bank")],
              [sg.Button("Pohalim Bank")], [sg.Text("Written By Sapir Paley , ID : 314971458")]]
    # Create the window
    window = sg.Window("Cheque Identifier For Dr.Amir Hendlman ", layout)

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        match event:
            case "Leumi Bank":
                Chosed_Bank = "Leumi_cheque.jpg"
                break
            case "BinLeumi Bank":
                Chosed_Bank = "BinLeumi_Bank_Cheque.jpg"
                break
            case "Discount Bank":
                Chosed_Bank = "Discount_bank_cheque.jpg"
                break
            case "Mizrahi Bank":
                Chosed_Bank = "Mizrahi_bank_cheque.jpg"
                break
            case "Pohalim Bank":
                Chosed_Bank = "Pohalim_bank_cheque.jpeg"
                break
        if event == sg.WIN_CLOSED:
            break
    window.close()
    return Chosed_Bank


def Extract_string_From_Image(Proccessed_img):
    text = pytesseract.image_to_string(Proccessed_img)
    return text

def get_grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def Remove_Noise(image):
    return cv2.medianBlur(image,5)

def thresholding(image):
    return cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def Find_Bank_Sum (Image,template_Sum,Bank_Cheque):
    img = cv2.imread(Bank_Cheque, 0)
    img = thresholding(img)
    img = Remove_Noise(img)
    template = cv2.imread(template_Sum, 0)
    template = thresholding(template)
    template = Remove_Noise(template)
    w, h = template.shape[::-1]
    # Apply template Matching = 'cv2.TM_SQDIFF' = '0'
    res = cv2.matchTemplate(img, template, 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    Result_Boundries_img = cv2.rectangle(Image, top_left, bottom_right, (0, 0, 255), 2)
    cv2.imshow("The Bank Symbole:", Result_Boundries_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    return Result_Boundries_img
def Find_Bank_Name(Image,template_Name,Bank_Cheque):
    img = cv2.imread(Bank_Cheque, 0)
    img = thresholding(img)
    img = Remove_Noise(img)
    template = cv2.imread(template_Name, 0)
    template = thresholding(template)
    template = Remove_Noise(template)
    w, h = template.shape[::-1]
    # Apply template Matching = 'cv2.TM_SQDIFF' = '0'
    res = cv2.matchTemplate(img, template, 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    Result_Boundries_img = cv2.rectangle(Image, top_left, bottom_right, 255, 2)
    cv2.imshow("The Bank Symbole:", Result_Boundries_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    return Result_Boundries_img
def Find_Bank_Symbole(Bank_Cheque,template_symbole):
    img = cv2.imread(Bank_Cheque, 0)
    template = cv2.imread(template_symbole, 0)
    img = thresholding(img)
    img = Remove_Noise(img)
    template=thresholding(template)
    template=Remove_Noise(template)
    w, h = template.shape[::-1]
    # Apply template Matching = 'cv2.TM_SQDIFF' = '0'
    res = cv2.matchTemplate(img, template, 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    original_img = cv2.imread(Bank_Cheque)
    Result_Boundries_img = cv2.rectangle(original_img, top_left, bottom_right, 255, 2)
    cv2.imshow("The Bank Symbole:", Result_Boundries_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    return Result_Boundries_img

def Init(Bank_Cheque):

    match Bank_Cheque:

        case "BinLeumi_Bank_Cheque.jpg":
                template_symbole = "BinLeumi_Bank_symbole.jpg"
                template_Name= "BinLeumi_Bank_Name.jpg"
                template_sum = "BinLeumi_Bank_sum.jpg"
        case "Discount_bank_cheque.jpg":
              template_symbole = "Discount_bank_symbole.jpg"
              template_Name = "Discount_bank_symbole.jpg"
              template_sum = "Discount_bank_sum.jpg"
        case "Leumi_cheque.jpg":
            template_symbole = "Leumi_cheque_Symbole.png"
            template_Name = "Leumi_cheque_Name.png"
            template_sum = "leumi_template_sum.png"
        case "Mizrahi_bank_cheque.jpg":
            template_symbole = "Mizrahi_bank_cheque_Symbole.png"
            template_Name = "Mizrahi_bank_cheque_Name.png"
            template_sum = "Mizrahi_bank_Sum.jpg"
        case "Pohalim_bank_cheque.jpeg":
            template_symbole = "Pohalim_bank_symbole.jpg"
            template_Name = "Pohalim_bank_Name.jpg"
            template_sum = "Pohalim_bank_Sum.jpg"

    result= Find_Bank_Symbole(Bank_Cheque, template_symbole)
    result = Find_Bank_Name(result,template_Name,Bank_Cheque)
    result = Find_Bank_Sum(result,template_sum,Bank_Cheque)
    layout = [[sg.Text("Thank you, Would you like to upload a different cheque ? \n")], [sg.Button("YES!")],[sg.Button("GoodBye!")],[sg.Text("Written By Sapir Paley , ID : 314971458")]]
    # Create the window
    window = sg.Window("Cheque Identifier For Dr.Amir Hendlman ", layout)
    event, values = window.read()
    if event == "YES!":
        window.close()
        Chosed_Bank = Start_GUI()
        Init(Chosed_Bank)
    else:
        window.close()


Chosed_Bank=Start_GUI()
Init(Chosed_Bank)

'''''
#Find symboles
img = cv2.imread('Leumi_cheque.jpg',0)
#img= get_grayscale(img)
img = thresholding(img)
img=Remove_Noise(img)
img2 = img.copy()
template = cv2.imread('Leumi_cheque_Symbole.png',0)
template = thresholding(template)
template=Remove_Noise(template)
w, h = template.shape[::-1]

# Apply template Matching = 'cv2.TM_SQDIFF' = '0'
res = cv2.matchTemplate(img,template,0)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
original_img = cv2.imread('Leumi_cheque.jpg')
Result_Boundries_img = cv2.rectangle(original_img,top_left, bottom_right, 255, 2)
cv2.imshow("The Bank Symbole:",Result_Boundries_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
#cv2.imwrite("New_Saved_img",result)


#Find Bank Name
#img = cv2.imread('Mizrahi_bank_cheque.jpg',0)
#img= get_grayscale(img)
img = thresholding(img)
img=Remove_Noise(img)
img2 = img.copy()
template = cv2.imread('Leumi_cheque_Name.png',0)
template = thresholding(template)
template=Remove_Noise(template)
w, h = template.shape[::-1]

# Apply template Matching = 'cv2.TM_SQDIFF' = '0'
res = cv2.matchTemplate(img,template,0)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
#original_img = cv2.imread('Leumi_cheque.jpg')
Result_Boundries_img = cv2.rectangle(Result_Boundries_img,top_left, bottom_right, 255, 2)
cv2.imshow("The Bank Name:",Result_Boundries_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()


#Find Bank sum
#img = cv2.imread('cheque_Example.jpg',0)
#img= get_grayscale(img)
img = thresholding(img)
img=Remove_Noise(img)
img2 = img.copy()
template = cv2.imread('leumi_template_sum.png',0)
template = thresholding(template)
template=Remove_Noise(template)
w, h = template.shape[::-1]

# Apply template Matching = 'cv2.TM_SQDIFF' = '0'
res = cv2.matchTemplate(img,template,0)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
original_img = cv2.imread('cheque_Example.jpg')
Result_Boundries_img = cv2.rectangle(Result_Boundries_img,top_left, bottom_right, (0, 0, 255), 2)
cv2.imshow("The Bank Sum",Result_Boundries_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
'''


