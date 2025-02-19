
import cv2
import pytesseract
import pandas as pd

def extract_text_to_dataframe(image_path):
    # Bild laden und in Graustufen umwandeln
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # OCR ausführen und Ergebnisse in DataFrame speichern
    ocr_data = pytesseract.image_to_data(binary_image, config='--psm 6')
    ocr_rows = []
    
    for n, row in enumerate(ocr_data.splitlines()):
        if n != 0:
            row_data = row.split()
            if len(row_data) == 12:
                x, y, w, h, text = int(row_data[6]), int(row_data[7]), int(row_data[8]), int(row_data[9]), row_data[11]
                if text.strip():  # Leere Ergebnisse ignorieren
                    ocr_rows.append({"Text": text, "x": x, "y": y, "Width": w, "Height": h})

    df = pd.DataFrame(ocr_rows)

    # 1. Sortiere nach x-Werten, um Spalten zu bestimmen
    df = df.sort_values(by="x")

    # Gruppiere nach ähnlichen x-Werten (flexibler Schwellenwert für Spaltenbereiche)
    threshold_x = 20  # Toleranz für x-Abstände innerhalb einer Spalte
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

    # 2. Sortiere innerhalb jeder Spalte nach y-Werten
    sorted_columns = []
    for col_df in column_groups:
        col_df_sorted = col_df.sort_values(by="y").reset_index(drop=True)
        sorted_columns.append(col_df_sorted["Text"])

    # 3. Füge alle Spalten zu einem DataFrame zusammen
    table_df = pd.concat(sorted_columns, axis=1).fillna("")
    table_df.columns = [f"Spalte_{i+1}" for i in range(len(sorted_columns))]

    return table_df


def main(image_path, output_excel_path):
    # Extrahiere die Daten und speichere sie in eine Excel-Datei
    table_df = extract_text_to_dataframe(image_path)
    print(table_df)
    table_df.to_excel(output_excel_path)
    print(f"Excel-Datei erfolgreich gespeichert unter {output_excel_path}")


# # Teste das Skript
# image_path = "dein_bild.png"
# output_excel_path = "extrahierte_tabelle.xlsx"
# set path to tesseract
pytesseract.pytesseract.tesseract_cmd = "C://Program Files//Tesseract-OCR//tesseract.exe"

# set path to image file
image_path = "C:/Users/49176/OneDrive/Desktop/OCR/greek_letters.png"

# set path to store the processed image
#temp_path = "C:/Users/49176/OneDrive/Desktop/OCR/processed_images/table 2.png"

# set path to csv file for results
output_excel_path = "C:/Users/49176/OneDrive/Desktop/OCR/letters_2.xlsx"
main(image_path, output_excel_path)