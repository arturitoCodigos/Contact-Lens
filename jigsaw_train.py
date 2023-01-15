import os
import numpy as np
import tensorflow as tf
from cv2 import imread
from sklearn.model_selection import train_test_split

def toInt(s):
    return int(s)

def readDir(path="./imgs", info=True):
    features, labels = [], []
    files = os.listdir(path)
    if info:
        print(f"Sao {len(files)} arquivos a serem lidos!")
        i = 0
    for img_name in files:
        features.append(imread(os.path.join(path, img_name)))
        img_name = os.path.splitext(img_name)[0]
        labels.append(list(map(toInt, img_name.split("-"))))
        if info:
            if (i%100 == 0):
                print(f"Estamos no arquivo numero {i}!")
            i+=1
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Carregando o dataset
    features, labels = readDir()

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=23, shuffle=True)

    # Neural Net em si

    # Modelo 1 --> Soprado da documentação do keras

    inputs = tf.keras.layers.Input(shape=(300,300,3))
    pre_treinada_saida = inputs
    neuralNet = tf.keras.applications.EfficientNetB0(weights="imagenet", input_tensor=pre_treinada_saida, include_top=False)

    neuralNet.trainable = True # O modelo base nao treina

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


