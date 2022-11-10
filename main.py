import os
import numpy as np
from cv2 import imread

def read_dir(path):
    result = []
    for img in os.listdir(path):
        result.append(list(imread(os.path.join(path, img))))
    return result

def create_dataset(folder_path):
    colored, normal, transparent = [], [], []
    
    # Iterar por cada uma das 100 pastas
    # Cada pasta contem subpastas 'Colored', 'Normal' e 'Transparent'
    for paste in os.listdir(folder_path):
        path1 = os.path.join(folder_path, paste, "Colored")
        path2 = os.path.join(folder_path, paste, "Normal")
        path3 = os.path.join(folder_path, paste, "Transparent")
        colored += read_dir(path1)
        normal += read_dir(path2)
        transparent += read_dir(path3)
    
    return np.array(colored), np.array(normal), np.array(transparent)

if __name__ == "__main__":
    c, n, t = create_dataset("/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB/Cogent Scanner")
    print(c.shape)
    print(n.shape)
    print(t.shape)








        
