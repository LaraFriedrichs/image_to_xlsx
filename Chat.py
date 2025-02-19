import cv2
import pytesseract
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def process_image(image_path):
    # Bild laden und vorverarbeiten
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bild binarisieren
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontale und vertikale Linien erkennen
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Linien kombinieren und Konturen erkennen
    combined_lines = detect_horizontal + detect_vertical
    contours, _ = cv2.findContours(combined_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bounding-Boxen aus Konturen extrahieren
    table_cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        table_cells.append((x, y, w, h))

    return image, table_cells


def extract_text_from_cells(image, table_cells):
    # Textinformationen aus OCR sammeln
    data = []
    for (x, y, w, h) in table_cells:
        roi = image[y:y + h, x:x + w]
        text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        if text:  # Nur relevante Inhalte speichern
            data.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})

    return pd.DataFrame(data)


def cluster_text(df):
    # DBSCAN für Zeilen-Clustering basierend auf y-Koordinaten
    if df.empty:
        print("Keine Daten extrahiert.")
        return pd.DataFrame()

    clustering = DBSCAN(eps=15, min_samples=1, metric='euclidean').fit(df[['y']])
    df['row_cluster'] = clustering.labels_

    # Zeilenweise sortieren und innerhalb jeder Zeile x-Werte verwenden
    table_data = []
    for row_label in sorted(df['row_cluster'].unique()):
        row_data = df[df['row_cluster'] == row_label].sort_values(by='x')
        row_texts = list(row_data['Text'])
        table_data.append(row_texts)

    return pd.DataFrame(table_data)


def save_table_to_excel(df, output_excel_path):
    # Excel speichern
    df.to_excel(output_excel_path, index=False, header=False)
    print(f"Excel-Datei erfolgreich unter {output_excel_path} gespeichert.")


def main(image_path, output_excel_path):
    # Schritt 1: Bild verarbeiten und Zellen erkennen
    image, table_cells = process_image(image_path)

    # Schritt 2: Text aus den erkannten Zellen extrahieren
    text_df = extract_text_from_cells(image, table_cells)

    # Schritt 3: Text in tabellarische Struktur bringen
    table_df = cluster_text(text_df)
    print(table_df)
    # Schritt 4: Excel speichern
    #save_table_to_excel(table_df, output_excel_path)


# Teste das Skript
# image_path = "dein_bild.png"
# output_excel_path = "extrahierte_tabelle.xlsx"

# set path to tesseract
pytesseract.pytesseract.tesseract_cmd = "C://Program Files//Tesseract-OCR//tesseract.exe"

# set path to image file
image_path = "C:/Users/49176/OneDrive/Desktop/OCR/greek_letters.png"

# set path to store the processed image
#temp_path = "C:/Users/49176/OneDrive/Desktop/OCR/processed_images/table 2.png"

# set path to csv file for results
output_excel_path = "C:/Users/49176/OneDrive/Desktop/OCR/letters.xlsx"
main(image_path, output_excel_path)