import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load image with open cv
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if image is None:
        print("Fehler: Bild konnte nicht geladen werden.")
    else:
        print("Bild erfolgreich geladen.")
    return image

# display image
def display_image(image, title="Image"):
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# produce a grayscale image
def grayscale_image(image):  
    grayscaled_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    grayscaled_image = cv2.equalizeHist(grayscaled_image)
    return grayscaled_image

# create a binary image
def binary_image(image,thresh1,thresh2,threshtype):
    #thresh, bin_image = cv2.threshold(image, 170, 240, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    #thresh, bin_image = cv2.threshold(image, 120, 170, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    #thresh, bin_image = cv2.threshold(image, 170, 240, cv2.THRESH_BINARY)
    thresh, bin_image = cv2.threshold(image,thresh1,thresh2,threshtype)     
    return bin_image

# dilate image
def dilate_image(image):
    image = cv2.bitwise_not(image)
    kernal = np.ones((2,1),np.uint8)
    image = cv2.dilate(image, kernal, iterations = 1)
    image = cv2.bitwise_not(image)
    return image

# erode image
def erode_image(image):
    image = cv2.bitwise_not(image)
    kernal = np.ones((2,1),np.uint8)
    image = cv2.erode(image, kernal, iterations = 1)
    image = cv2.bitwise_not(image)
    return image

# blur image
def blur_image(image):
    kernal=(1,1)
    blurred_image = cv2.GaussianBlur(image, kernal, sigmaX=0)
    return blurred_image

# remove horizontal lines 
def remove_horizontal_lines(image):
    image = cv2.bitwise_not(image)
    #horizontal_kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 1))
    horizontal_kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(image).shape[1]//30, 1))
    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernal)
    no_horizontal_lines_image = cv2.subtract(image, horizontal_lines)
    no_horizontal_lines_image = cv2.bitwise_not(no_horizontal_lines_image)
    return no_horizontal_lines_image

# remove vertical lines 
def remove_vertical_lines(image):
    image = cv2.bitwise_not(image)
    #vertical_kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(image).shape[1]//30))
    vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
    no_vertical_lines_image = cv2.subtract(image, vertical_lines)
    no_vertical_lines_image = cv2.bitwise_not(no_vertical_lines_image)
    return no_vertical_lines_image

