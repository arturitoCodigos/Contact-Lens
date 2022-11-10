import os
import numpy as np
from sklearn.model_selection import train_test_split
from cv2 import imread

def read_dir(path):
    result = []
    for img in os.listdir(path):
        result.append(list(imread(os.path.join(path, img))))
    return result

def create_dataset(folder_path, info=False):
    colored, normal, transparent = [], [], []

    # Informativo do load
    i = 0
    total = len(os.listdir(folder_path))

    # Iterar por cada uma das 100 pastas
    # Cada pasta contem subpastas 'Colored', 'Normal' e 'Transparent'
    for paste in os.listdir(folder_path):
        path1 = os.path.join(folder_path, paste, "Colored")
        path2 = os.path.join(folder_path, paste, "Normal")
        path3 = os.path.join(folder_path, paste, "Transparent")
        colored += read_dir(path1)
        normal += read_dir(path2)
        transparent += read_dir(path3)

        # Info
        i+=1
        if (i % 100 == 0):
            print(f"{(i/total)*100}% concluido!")
    
    return np.array(colored), np.array(normal), np.array(transparent)

# 1 -> Colored
# 2 -> Normal
# 3 -> Transparent 
def lens_dataset(folder_path="/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB/Cogent Scanner"):
    c, n, t = create_dataset(folder_path)
    features = np.concatenate((c, n, t))
    labels = np.array([1 for _ in range(c.shape[0])] + [2 for _ in range(n.shape[0])] + [3 for _ in range(t.shape[0])])
    return features, labels

if __name__ == "__main__":
    # Carregando o dataset
    features, labels = lens_dataset()
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=23, shuffle=True)
    
    # Neural Net em si
    








        
