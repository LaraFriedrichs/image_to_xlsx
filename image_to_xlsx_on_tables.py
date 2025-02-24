import cv2
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import load_image
from functions import display_image
from functions import grayscale_image
from functions import dilate_image
from functions import erode_image
from functions import blur_image
from functions import binary_image
from functions import remove_horizontal_lines
from functions import remove_vertical_lines

##############################################################################################################################################

# Changes

# 1. chosse weather lines should be removed during the preprocessing or not
remove_lines = True

# 2. set paths to your image and xlsx and csv file for results
image_path = "C:/Users/49176/OneDrive/Desktop/OCR/OCR/images/4.4.2_no_notes.png"
output_excel_path = "C:/Users/49176/OneDrive/Desktop/OCR/OCR/results/4.4.2_no_notes.xlsx"
output_csv_path = "C:/Users/49176/OneDrive/Desktop/OCR/OCR/results/4.4.2_no_notes.csv"

# 3. set up tesseract
pytesseract.pytesseract.tesseract_cmd = "C://Program Files//Tesseract-OCR//tesseract.exe"

# 4. set x threshold for col and row seperation

threshold_x=10
threshold_y=10

# 5. set the thresholding type
#threshtype = "OTSU"
threshtype = "BINARY"
#threshtype = "BINARY_manuel"

# 6. set thresh1 and thresh 2 for threshtype = BINARY_manuel

thresh1=170 # 0–255
thresh2=255 # 0-255 255 recommended

# 7. select tesseract engine and page segmentation mode 

# oem 1 or 3 and psm 1,3,4,6,11,12 are good for tables

custom_config = r'--oem 3 --psm 11 -l grc+eng -c tessedit_char_whitelist="0123456789IVXLCDM¹²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉⁻abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ𝑎𝑏𝑐𝑑𝑒𝑓𝑔ℎ𝑖𝑗𝑘𝑙𝑚𝑛𝑜𝑝𝑞𝑟𝑠𝑡𝑢𝑣𝑤𝑥𝑦𝑧𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐽𝐾𝐿𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑊𝑋𝑌𝑍αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ.--_/=()[]{}~.%,:;<>+*" -c tessedit_preserve_interword_spaces=1 -c preserve_interword_spaces=1x1'


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
####################################################################################################################################
# image preprocessing

image = load_image(image_path)
gray_image = grayscale_image(image)
dilated_image = dilate_image(gray_image)
blurred_image = blur_image(dilated_image)
image_without_hlines = remove_horizontal_lines(blurred_image)
image_without_vlines = remove_vertical_lines(image_without_hlines)
eroded_image = erode_image(image_without_vlines)
if remove_lines == True:
    if threshtype == "BINARY":
        bin_image = cv2.adaptiveThreshold(eroded_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif threshtype == "OTSU":
        bin_image = binary_image(eroded_image,170,255,cv2.THRESH_OTSU)
    elif threshtype == "Binary_manuel":    
        bin_image = binary_image(eroded_image,thresh1, cv2.THRESH_BINARY)
    else: 
        print("Please choose the type of Threshold you want to use")
elif remove_lines == False:
    if threshtype == "BINARY":
        bin_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif threshtype == "OTSU":
        bin_image = binary_image(gray_image,170,255,cv2.THRESH_OTSU)
    elif threshtype == "Binary_manuel":    
        bin_image = binary_image(gray_image,thresh1, cv2.THRESH_BINARY)
    else: 
        print("Please choose the type of Threshold you want to use")
else: 
    print("Please choose weather you want to remove the lines or not")

# OCR 

ocr_data = pytesseract.image_to_data(bin_image, config=custom_config)
rows = []
# for n, row in enumerate(ocr_data.splitlines()):
#     if n == 0:
#         continue  # Header-Zeile überspringen
#     row_data = row.split()
#     if len(row_data) == 12:
#         x, y, w, h, text = map(int, row_data[6:10]) + [row_data[11]]
#         if text.strip():
#             rows.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})

for n, row in enumerate(ocr_data.splitlines()):
  if n != 0:
    row_data = row.split()
    if len(row_data) == 12:
        x, y, w, h, text = int(row_data[6]), int(row_data[7]), int(row_data[8]), int(row_data[9]), row_data[11]
        if text.strip():  # Leere Ergebnisse ignorieren
            rows.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})

df = pd.DataFrame(rows)
df = df.sort_values(by="x")
column_groups = []
current_group = []
last_x = -1

# Gruppiere Spalten anhand der x-Werte
for _, row in df.iterrows():
    if last_x == -1 or abs(row["x"] - last_x) <= threshold_x:
        current_group.append(row)
    else:
        column_groups.append(pd.DataFrame(current_group))
        current_group = [row]
    last_x = row["x"]

if current_group:
    column_groups.append(pd.DataFrame(current_group))
# Erstelle eine Liste aller einzigartigen y-Werte für Zeilenabgleich
all_y_values = sorted(set(df["y"]))
# Fülle die Spalten so, dass leere Felder nicht zu Verschiebungen führen
aligned_columns = []
for col_df in column_groups:
    col_df = col_df.sort_values(by="y").reset_index(drop=True)
    aligned_col = []
    
    # Finde die nächstgelegenen y-Werte und ordne Text den passenden Zeilen zu
    for y in all_y_values:
        matching_texts = col_df[(col_df["y"] >= y - threshold_y) & (col_df["y"] <= y + threshold_y)]
        aligned_col.append(matching_texts["Text"].iloc[0] if not matching_texts.empty else "")
    aligned_columns.append(aligned_col)
# Baue DataFrame mit ausgerichteten Zeilen auf
table_df = pd.DataFrame(aligned_columns).T.fillna("")
#new_table_df = table_df.loc[df.shift() != table_df].dropna().reset_index(drop=True)
new_table_df = table_df.loc[(table_df != table_df.shift()).any(axis=1)].reset_index(drop=True) # diese reihe ist verantwortlich dafür das thresh binary nicht mehr funktioniert

print(new_table_df)
new_table_df.to_excel(output_excel_path)
print(f"Saved file as {output_excel_path}")

new_table_df.to_csv(output_csv_path)
print(f"Saved file as {output_csv_path}")

############# simple OCR ############

# # OCR
# ocr_data = pytesseract.image_to_data(bin_image, config=custom_config)
# # store results in a dataframe
# ocr_rows = []
# for n, row in enumerate(ocr_data.splitlines()):
#     if n != 0:
#         row_data = row.split()
#         if len(row_data) == 12:
#             x, y, w, h, text = int(row_data[6]), int(row_data[7]), int(row_data[8]), int(row_data[9]), row_data[11]
#             if text.strip():  # Leere Ergebnisse ignorieren
#                 ocr_rows.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})
# df = pd.DataFrame(ocr_rows)
# # define columns
# df = df.sort_values(by="x")
# column_groups = []
# current_group = []
# last_x = -1
# for _, row in df.iterrows():
#     if last_x == -1 or abs(row["x"] - last_x) <= threshold_x:
#         current_group.append(row)
#     else:
#         column_groups.append(pd.DataFrame(current_group))
#         current_group = [row]
#     last_x = row["x"]
# if current_group:
#     column_groups.append(pd.DataFrame(current_group))
# # sort text in columns to reproduce the rows
# sorted_columns = []
# for col_df in column_groups:
#     col_df_sorted = col_df.sort_values(by="y").reset_index(drop=True)
#     sorted_columns.append(col_df_sorted["Text"])
# # store results in a DataFrame
# table_df = pd.concat(sorted_columns, axis=1).fillna("")
# table_df.columns = [f"column_{i+1}" for i in range(len(sorted_columns))]
# print(table_df)
# table_df.to_excel(output_excel_path)
# print(f"Saved file as {output_excel_path}")

################################# different approach #################################

# """Führt OCR aus und speichert Ergebnisse mit Positionen in einem DataFrame."""
# ocr_data = pytesseract.image_to_data(bin_image, config=custom_config)
# rows = []
# for n, row in enumerate(ocr_data.splitlines()):
#   if n != 0:
#      row_data = row.split("\t")
#      #print(row_data)
#      if len(row_data) == 12:
#         x, y, w, h, text = int(row_data[6]), int(row_data[7]), int(row_data[8]), int(row_data[9]), row_data[11]
#         if text.strip():
#             rows.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})
# df=pd.DataFrame(rows)

# # #for i in range(1,len(df)):
# # if df["Height"].item < 10:
# #    df.drop()
# # print(df)
# """Richtet erkannte Wörter in einer Tabelle aus, sodass Spalten und Zeilen korrekt ausgerichtet sind.
#    Mehrfacheinträge in derselben Spalte werden zusammengefasst."""

# # **1. Einzigartige x- und y-Werte sammeln**
# unique_x_values = sorted(set(df["x"]))
# unique_y_values = sorted(set(df["y"]))
# # **2. Erstelle ein leeres Raster für die Tabelle**
# aligned_data = {y: {x: "" for x in unique_x_values} for y in unique_y_values}
# # **3. Füge erkannte Wörter in das Raster ein (nächstgelegene x- und y-Werte)**
# for _, row in df.iterrows():
#     closest_y = min(unique_y_values, key=lambda y: abs(y - row["y"]))
#     closest_x = min(unique_x_values, key=lambda x: abs(x - row["x"]))
    
#     # Falls in der Zelle schon ein Wert existiert, neuen Wert anhängen
#     if aligned_data[closest_y][closest_x]:
#         aligned_data[closest_y][closest_x] += " " + row["Text"]  # Trennzeichen " "
#     else:
#         aligned_data[closest_y][closest_x] = row["Text"]
# # **4. Erstelle die Tabelle als DataFrame**
# aligned_rows = []
# for y in unique_y_values:
#     aligned_rows.append([aligned_data[y][x] for x in unique_x_values])
# table_df = pd.DataFrame(aligned_rows, columns=[f"column_{i+1}" for i in range(len(unique_x_values))])
# print(table_df.dropna())


