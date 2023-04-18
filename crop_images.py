import os
import csv
from cv2 import imread, imwrite
from math import floor

# 9, 10, 11

def crop(base, path, x, y, r):
    try:
        img = imread(os.path.join(base, path))
        w, h, _ = img.shape
        imwrite(os.path.join("/media/work/arthurcosta/Contact-Lens/test_output/media/work/datasets/contact-lens/crop/IIITD", path), 
                img[max(0, floor(y-r-(r*0.1))):floor(y+r+(r*0.1)), 
                    max(0, floor(x-r-(r*0.1))):floor(x+r+(r*0.1))])
        print("Done")
    except Exception as e:
        print("ERRO NA FUNCAO DE CROP\n\n\n\n")
        print(e)

def main(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x, y, r = int(row[9]), int(row[10]), int(row[11])
            crop("/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB", row[0]+".bmp", x, y, r)


if __name__ == "__main__":
    main("/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB/CogentAnnotationTrain.csv")
    main("/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB/CogentAnnotationTest.csv")
