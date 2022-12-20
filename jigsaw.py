import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from cv2 import imread, resize
from random import sample, seed

from PIL import Image

seed(10) # Reproducibilidade dos experimentos!

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

def read_dir(path, multiplier=1):
    imgs, lbls = [], []
    for img in os.listdir(path):
        img = imread(os.path.join(path, img))
        res = resize(img, dsize=(300, 300))

        for _ in range(multiplier):
            # Jigsaw shuffle
            label = sample(range(0,9), k=9)
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
    return imgs, lbls

def create_dataset(folder_path, info=False):
    colored, normal, transparent = [], [], []
    colored_lbl, normal_lbl, transparent_lbl = [], [], []

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

        colored     += p1[0]
        normal      += p2[0]
        transparent += p3[0]

        colored_lbl     += p1[1]
        normal_lbl      += p2[1]
        transparent_lbl += p3[1]

        # Info
        i+=1
        if (i % 100 == 0):
            print(f"{(i/total)*100}% concluido!")
    
    return (np.array(colored), np.array(normal), np.array(transparent)), (np.array(colored_lbl), np.array(normal_lbl), np.array(transparent_lbl))

# 1 -> Colored
# 2 -> Normal
# 3 -> Transparent 
def lens_dataset(folder_path="/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB/Cogent Scanner"):
    features, labels = create_dataset(folder_path)
    c, n, t = features
    c_lbl, n_lbl, t_lbl = labels

    f_conc = np.concatenate((c, n, t))
    l_conc = np.concatenate((c_lbl, n_lbl, t_lbl))
    return f_conc, l_conc

if __name__ == "__main__":
    # Carregando o dataset
    features, labels = lens_dataset()

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=23, shuffle=True)

    # Neural Net em si

    # Modelo 1 --> Soprado da documentação do keras

    inputs = tf.keras.layers.Input(shape=(300,300,3))
    pre_treinada_saida = inputs
    neuralNet = tf.keras.applications.EfficientNetB0(weights="imagenet", input_tensor=pre_treinada_saida, include_top=False)

    neuralNet.trainable = False # O modelo base nao treina

    pre_treinada_saida = tf.keras.layers.GlobalAveragePooling2D()(neuralNet.output)
    pre_treinada_saida = tf.keras.layers.BatchNormalization()(pre_treinada_saida)
    pre_treinada_saida = tf.keras.layers.Dropout(0.2)(pre_treinada_saida)
    out = tf.keras.layers.Dense(9, activation="softmax")(pre_treinada_saida)  # Output

    # Construção do modelo
    neuralNet = tf.keras.Model(inputs, out)

    #neuralNet.summary() # Resumo da rede

    neuralNet.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
    
    neuralNet.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test), verbose=1)
    neuralNet.save("./contact-lens-model-effnetb0")
