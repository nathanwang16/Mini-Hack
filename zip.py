import csv
import zipfile
import os


# File paths
txt_file_path = '/Users/xiaoyuwang/Desktop/Mini-Hack/2983predictions.txt'  # Replace with your .txt file
csv_file_path = '/Users/xiaoyuwang/Desktop/Mini-Hack/submit/predictions.csv'
zip_file_path = '/Users/xiaoyuwang/Desktop/Mini-Hack/submit/predictions.zip'

import csv
import zipfile
import os

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt:
        lines = txt.readlines()

    with open(csv_file, 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        for line in lines:
            # Split the line into columns based on spaces or commas
            writer.writerow(line.strip().split())

def zip_file(file_to_zip, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zip_f:
        zip_f.write(file_to_zip, os.path.basename(file_to_zip))
    print(f"{file_to_zip} has been zipped into {zip_name}")



# Convert .txt to .csv
txt_to_csv(txt_file_path, csv_file_path)

# Compress .csv into .zip
zip_file(csv_file_path, zip_file_path)