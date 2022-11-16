import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from cv2 import imread, resize

def read_dir(path):
    result = []
    for img in os.listdir(path):
        img = imread(os.path.join(path, img))
        res = resize(img, dsize=(224, 224))
        result.append(list(res))
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
    labels = np.array([[1, 0, 0] for _ in range(c.shape[0])] + [[0, 1, 0] for _ in range(n.shape[0])] + [[0, 0, 1] for _ in range(t.shape[0])]) # One-Hot
    return features, labels

if __name__ == "__main__":
    # Carregando o dataset
    features, labels = lens_dataset()
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=23, shuffle=True)
    
    # Neural Net em si

    # Camada para augmentação de imagem
    img_augmentation = Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )

    # Modelo 1 --> Soprado da documentação do keras

    inputs = tf.keras.layers.Input(shape=(224,224,3))
    pre_treinada_saida = img_augmentation(inputs)
    neuralNet = tf.keras.applications.EfficientNetB0(weights="imagenet", input_tensor=pre_treinada_saida, include_top=False)

    neuralNet.trainable = False # O modelo base nao treina

    pre_treinada_saida = tf.keras.layers.GlobalAveragePooling2D()(neuralNet.output)
    pre_treinada_saida = tf.keras.layers.BatchNormalization()(pre_treinada_saida)
    pre_treinada_saida = tf.keras.layers.Dropout(0.2)(pre_treinada_saida)
    out = tf.keras.layers.Dense(3, activation="softmax")(pre_treinada_saida)  # Output

    # Construção do modelo
    neuralNet = tf.keras.Model(inputs, out)

    #neuralNet.summary() # Resumo da rede

    neuralNet.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
    
    neuralNet.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test), verbose=1)
    neuralNet.save("./contact-lens-model-effnetb0")
