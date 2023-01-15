import os
import numpy as np
import tensorflow as tf
from cv2 import imread
from sklearn.model_selection import train_test_split

def toInt(s):
    return int(s)

def readDir(path="./imgs"):
    features, labels = [], []
    files = os.listdir(path)
    for img_name in files:
        features.append(imread(os.path.join(path, img_name)))
        img_name = os.path.splitext(img_name)[0]
        labels.append(list(map(toInt, img_name.split("-"))))
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Carregando o dataset
    features, labels = readDir()

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=23, shuffle=True)

    # Neural Net em si

    # Modelo 1 --> Soprado da documentação do keras

    inputs = tf.keras.layers.Input(shape=(300,300,3))
    pre_treinada_saida = inputs
    neuralNet = tf.keras.layers.Conv2D(64, 5, activation="relu", kernel_regularizer="l2")(pre_treinada_saida)
    neuralNet = tf.keras.layers.Conv2D(64, 5, activation="relu", kernel_regularizer="l2")(neuralNet)
    neuralNet = tf.keras.layers.Conv2D(64, 5, activation="relu", kernel_regularizer="l2")(neuralNet)
    neuralNet = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(neuralNet)
    neuralNet = tf.keras.layers.Conv2D(32, 3, activation="relu", kernel_regularizer="l2")(neuralNet)
    neuralNet = tf.keras.layers.Conv2D(32, 3, activation="relu", kernel_regularizer="l2")(neuralNet)
    neuralNet = tf.keras.layers.Conv2D(32, 3, activation="relu", kernel_regularizer="l2")(neuralNet)
    neuralNet = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(neuralNet)
    neuralNet = tf.keras.layers.GlobalAveragePooling2D()(neuralNet)
    neuralNet = tf.keras.layers.BatchNormalization()(neuralNet)
    neuralNet = tf.keras.layers.Dropout(0.2)(neuralNet)
    neuralNet = tf.keras.layers.Dense(128, activation="softmax", kernel_regularizer="l2")(neuralNet)
    neuralNet = tf.keras.layers.Dense(64, activation="softmax", kernel_regularizer="l2")(neuralNet)

    out = tf.keras.layers.Dense(9, activation="softmax")(neuralNet)  # Output

    # Construção do modelo
    neuralNet = tf.keras.Model(inputs, out)

    #neuralNet.summary() # Resumo da rede

    neuralNet.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
    
    neuralNet.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test), verbose=1)
    neuralNet.save("./contact-lens-model-custom-arch")


