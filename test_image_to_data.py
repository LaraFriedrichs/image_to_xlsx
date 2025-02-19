
import cv2
import numpy as np
import pytesseract
import pandas as pd
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import csv
import pandas as pd
from IPython.display import display

##############################################################################################################################################

# set path to tesseract
pytesseract.pytesseract.tesseract_cmd = "C://Program Files//Tesseract-OCR//tesseract.exe"

# set path to image file
image_path = "C:/Users/49176/OneDrive/Desktop/OCR/greek_letters.png"

# set path to store the processed image
#temp_path = "C:/Users/49176/OneDrive/Desktop/OCR/processed_images/table 2.png"

# set path to csv file for results
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
    thresh, bin_image = cv2.threshold(image, 170, 240, cv2.THRESH_BINARY |cv2.THRESH_OTSU)  #there is also cv2.THRESH_OTSU
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

def main(image_path,custom_config):
    image = load_image(image_path)
    grayscaled_image = grayscale_image(image)
    dilated_image = dilate_image(grayscaled_image)
    #blurred_image = blur_image(dilated_image)
    image_without_hlines = remove_horizontal_lines(dilated_image)
    eroded_image = erode_image(image_without_hlines)
    bin_image = binary_image(eroded_image)
    h_img,w_img =bin_image.shape
    boxes = pytesseract.image_to_data(bin_image, config=custom_config,lang='grc + eng')
    mean_height = np.mean([box[9] for box in boxes])
    rows = []
    for n, box in enumerate(boxes.splitlines()):
        if n != 0:
            box = box.split()
            if len(box) == 12:
                x,y,w,h = int(box[6]),int(box[7]),int(box[8]),int(box[9])
                image_with_boxes = cv2.rectangle(image,(x,y),(w+x,h+y),(0,0,255),1)
            if abs(box[9] - current_row[-1][1]) <= mean_height:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
                rows.append(current_row)
    return display_image(image_with_boxes)

def main_2(image_path, output_csv_path):
    # Bild laden und vorbereiten
    image = load_image(image_path)
    grayscaled_image = grayscale_image(image)
    bin_image = binary_image(grayscaled_image)

    # OCR-Erkennung mit Positionsdaten
    boxes = pytesseract.image_to_data(bin_image, config=custom_config)

    # Daten für den DataFrame sammeln
    data = []

    for n, box in enumerate(boxes.splitlines()):
        if n != 0:  # Skip header line
            box_data = box.split()
            if len(box_data) == 12:
                # Extrahiere Wort und Positionen
                x, y, w, h, text = int(box_data[6]), int(box_data[7]), int(box_data[8]), int(box_data[9]), box_data[11]
                data.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})

    # DataFrame erstellen und nach y und x sortieren
    df = pd.DataFrame(data)
    df = df.sort_values(by=["y", "x"]).reset_index(drop=True)
    display(df)

    # CSV-Datei speichern
    df.to_excel(output_excel_path, index=False)

    print(f"CSV exportiert nach: {output_csv_path}")



def image_to_table(image_path, output_excel_path):
    # Bild laden und verarbeiten
    image = load_image(image_path)
    grayscaled_image = grayscale_image(image)
    bin_image = binary_image(grayscaled_image)

    # OCR ausführen
    boxes = pytesseract.image_to_data(bin_image, config=custom_config)

    # OCR-Daten sammeln
    data = []
    for n, box in enumerate(boxes.splitlines()):
        if n != 0:
            box_data = box.split()
            if len(box_data) == 12:
                x, y, w, h, text = int(box_data[6]), int(box_data[7]), int(box_data[8]), int(box_data[9]), box_data[11]
                data.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})

    # DataFrame erstellen und nach y und x sortieren
    df = pd.DataFrame(data)
    df = df.sort_values(by=["x"]).reset_index(drop=True)

    # Clustering der Zeilen basierend auf y-Werten
    row_clusters = []
    threshold_y = 35 # Schwellenwert für y-Koordinaten (abhängig vom Bild)

    current_row = []
    last_y = None

    for _, row in df.iterrows():
        if last_y is None or abs(row["y"] - last_y) <= threshold_y:
            current_row.append(row)
        else:
            row_clusters.append(current_row)
            current_row = [row]
        last_y = row["y"]

    if current_row:
        row_clusters.append(current_row)

    # DataFrame für tabellarische Struktur erstellen
    table_data = []

    for row_cluster in row_clusters:
        row_text = []
        last_x = 0
        threshold_x = 70 # Schwellenwert für Spalten (abhängig vom Textabstand)

        for item in row_cluster:
            # Füge Platzhalter ein, wenn der Abstand zu groß ist
            while item["x"] > last_x + threshold_x:
                row_text.append(np.nan)
                last_x += threshold_x

            row_text.append(item["Text"])
            last_x = item["x"] + item["Width"]

        table_data.append(row_text)

    # DataFrame erstellen und NaN auffüllen
    table_df = pd.DataFrame(table_data).fillna("")
    display(table_df)
    # Excel speichern
    #table_df.to_excel(output_excel_path, index=False, header=False)

    #print(f"Excel-Datei erfolgreich unter {output_excel_path} gespeichert.")

image_to_table(image_path,output_excel_path)