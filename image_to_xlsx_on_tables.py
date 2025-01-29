import cv2
#from PIL import Image
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import display

##############################################################################################################################################

# set path to tesseract
pytesseract.pytesseract.tesseract_cmd = "C://Program Files//Tesseract-OCR//tesseract.exe"

# set path to image file
image_path = "C:/Users/49176/OneDrive/Desktop/OCR/greek_letters.png"

# set path to xlsx file for results
output_excel_path = "C:/Users/49176/OneDrive/Desktop/OCR/letters.xlsx"

############################################################################################################################################

# select tesseract engine and page segmentation mode 

custom_config = r'-l grc+eng -c preserve_interword_spaces=1x1 --oem 3 --psm 3' # psm 1,3,4,6,11,12 are good for tables
#custom_config = r' --oem 1 --psm 3'

# 1. --oem (OCR Engine Mode)

# The --oem parameter determines which OCR engine Tesseract should use. There are four modes:

#     --oem 0: Use only Tesseract's "Legacy" OCR engine. This is the older version of the engine and may be useful in certain specific cases.
#     --oem 1: Use only the new LSTM-based OCR engine. This is the neural network-based engine, which generally provides higher accuracy.
#     --oem 2: Combine the Legacy engine and the LSTM engine and select the best result.
#     --oem 3: Automatically select the engine based on the type of image (default and usually the best choice).

# The --psm parameter controls how Tesseract analyzes the page structure. There are 14 options:

#     --psm 0: Perform only orientation and script detection (OSD).
#     --psm 1: Automatic page segmentation with OSD.
#     --psm 2: Automatic page segmentation, but without OSD, and without assuming a single column.
#     --psm 3: Fully automatic page segmentation (default mode).
#     --psm 4: Segment the image into a single line of text.
#     --psm 5: Segment the image into a single vertical text block.
#     --psm 6: Segment the image into a single text block (useful for simple images without special structure).
#     --psm 7: Segment the image into a single text line containing only one line.
#     --psm 8: Segment the image into a single word.
#     --psm 9: Segment the image into a single word in a circular region (suitable for round texts or seals).
#     --psm 10: Segment the image into a single character.
#     --psm 11: Segment the image into a single block with variable text structure (useful for tables).
#     --psm 12: Segment the image into a single text block in vertical arrangement.
#     --psm 13: Segment the image into a single line of vertical text.

##############################################################################################################################################
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
    return grayscaled_image

# create a binary image
def binary_image(image):
    thresh, bin_image = cv2.threshold(image, 170, 240, cv2.THRESH_BINARY |cv2.THRESH_OTSU)  
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
    horizontal_kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(image).shape[1]//40, 1))
    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernal)
    no_horizontal_lines_image = cv2.subtract(image, horizontal_lines)
    no_horizontal_lines_image = cv2.bitwise_not(no_horizontal_lines_image)
    return no_horizontal_lines_image

# remove vertical lines 
def remove_vertical_lines(image):
    image = cv2.bitwise_not(image)
    #vertical_kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(image).shape[1]//40))
    vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
    no_vertical_lines_image = cv2.subtract(image, vertical_lines)
    no_vertical_lines_image = cv2.bitwise_not(no_vertical_lines_image)
    return no_vertical_lines_image

def extract_text_to_dataframe(image_path,custom_config):
    # load image
    image = cv2.imread(image_path)
    #create a gray image
    gray_image = grayscale_image(image)
    # remove the lines
    dilated_image = dilate_image(gray_image)
    #blurred_image = blur_image(dilated_image)
    image_without_hlines = remove_horizontal_lines(dilated_image)
    eroded_image = erode_image(image_without_hlines)
    #create binary image
    bin_image = binary_image(eroded_image)
    # OCR
    ocr_data = pytesseract.image_to_data(bin_image, config=custom_config)
    # store results in a dataframe
    ocr_rows = []
    for n, row in enumerate(ocr_data.splitlines()):
        if n != 0:
            row_data = row.split()
            if len(row_data) == 12:
                x, y, w, h, text = int(row_data[6]), int(row_data[7]), int(row_data[8]), int(row_data[9]), row_data[11]
                if text.strip():  # Leere Ergebnisse ignorieren
                    ocr_rows.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})
    df = pd.DataFrame(ocr_rows)
    # define columns
    df = df.sort_values(by="x")
    threshold_x = 20  
    column_groups = []
    current_group = []
    last_x = -1
    for _, row in df.iterrows():
        if last_x == -1 or abs(row["x"] - last_x) <= threshold_x:
            current_group.append(row)
        else:
            column_groups.append(pd.DataFrame(current_group))
            current_group = [row]
        last_x = row["x"]
    if current_group:
        column_groups.append(pd.DataFrame(current_group))
    # sort text in columns to reproduce the rows
    sorted_columns = []
    for col_df in column_groups:
        col_df_sorted = col_df.sort_values(by="y").reset_index(drop=True)
        sorted_columns.append(col_df_sorted["Text"])
    # store results in a DataFrame
    table_df = pd.concat(sorted_columns, axis=1).fillna("")
    table_df.columns = [f"column_{i+1}" for i in range(len(sorted_columns))]
    return table_df

def main(image_path, output_excel_path,custom_config):
    # custom_config = custom_config = r'-l grc+eng -c preserve_interword_spaces=1x1 --oem 3 --psm 3' 
    table_df = extract_text_to_dataframe(image_path,custom_config)
    print(table_df)
    table_df.to_excel(output_excel_path)
    print(f"Saved file as {output_excel_path}")

main(image_path, output_excel_path,custom_config)