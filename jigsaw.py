import os
import numpy as np
from cv2 import imread, resize, imwrite

three_way_partition = [[0, 1, 2, 
                        3, 4, 5, 
                        6, 7, 8],

                       [0, 2, 1,
                        3, 5, 4,
                        6, 8, 7],
                        
                       [1, 0, 2,
                        4, 3, 5,
                        7, 6, 8],
                        
                       [1, 2, 0,
                        4, 5, 3,
                        7, 8, 6],

                       [2, 0, 1,
                        5, 3, 4,
                        8, 6, 7],

                       [2, 1, 0,
                        5, 4, 3,
                        8, 7, 6]]

# ((linhas), (colunas))
limits = [((000, 100), (000, 100)),
          ((000, 100), (100, 200)),
          ((000, 100), (200, 300)),
          ((100, 200), (000, 100)),
          ((100, 200), (100, 200)),
          ((100, 200), (200, 300)),
          ((200, 300), (000, 100)),
          ((200, 300), (100, 200)),
          ((200, 300), (200, 300)),]

def toStr(s):
    return str(s)

def read_dir(path, multiplier=1):
    imgs, lbls = [], []
    for img in os.listdir(path):
        img = imread(os.path.join(path, img))
        res = resize(img, dsize=(300, 300))

        for arrangement in three_way_partition:
            # Jigsaw shuffle
            label = arrangement
            new_img = np.zeros((300, 300, 3), dtype=np.uint8)

            for i, i_label in enumerate(label):
                # Escrita na nova imagem
                l_min, l_max = limits[i][0] # Minimo e maximo pra linhas
                c_min, c_max = limits[i][1] # Minimo e maximo pra colunas

                # Leitura da imagem ja existente
                l_label_min, l_label_max = limits[i_label][0] # Minimo e maximo pra linhas
                c_label_min, c_label_max = limits[i_label][1] # Minimo e maximo pra colunas
                t = c_label_min
                for x in range(l_min, l_max): # Para cada linha da imagem original...
                    for y in range(c_min, c_max): # Para cada coluna da imagem original...
                        new_img[x][y] = res[l_label_min%l_label_max][c_label_min%c_label_max]
                        c_label_min += 1
                    l_label_min += 1
                    c_label_min = t
            imgs.append(list(new_img))
            lbls.append(list(label))
    return np.array(imgs, dtype=np.uint8), lbls

def create_dataset(folder_path, info=False):
    uuid = 0
    # Informativo do load
    i = 0
    total = len(os.listdir(folder_path))

    # Iterar por cada uma das 100 pastas
    # Cada pasta contem subpastas 'Colored', 'Normal' e 'Transparent'
    for paste in os.listdir(folder_path):
        path1 = os.path.join(folder_path, paste, "Colored")
        path2 = os.path.join(folder_path, paste, "Normal")
        path3 = os.path.join(folder_path, paste, "Transparent")

        p1 = read_dir(path1)
        p2 = read_dir(path2)
        p3 = read_dir(path3)

        iterables = [p1, p2, p3]

        for j in iterables:
            imgs, lbls = j
            print("Quantidade de imagens: ", len(imgs))
            for img, lbl in zip(imgs, lbls):
                lbl = str(uuid) + '=' + "-".join(list(map(toStr, lbl))) + ".png"
                imwrite("./three_way_dataset/" + lbl, img)
                uuid += 1
            
        # Info
        i+=1
        if (i % 100 == 0):
            print(f"{(i/total)*100}% concluido!")

if __name__ == "__main__":
    create_dataset("/media/work/datasets/contact-lens/crop/IIITD/Cogent Scanner")