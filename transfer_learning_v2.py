import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras import layers
import csv
from cv2 import imread, resize

def one_hot_encode_label(s):
    if s == "Colored":
        return [1, 0, 0]
    if s == "Normal":
        return [0, 1, 0]
    if s == "Transparent":
        return [0, 0, 1]
    print("Error on one hot encoding. Received value = ", s)
    exit(1)

def lens_dataset_csv(train_path="/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB/CogentAnnotationTrain.csv",
                     test_path ="/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB/CogentAnnotationTest.csv",
                     base_path ="/media/work/datasets/contact-lens/orig/IIITD_Contact_Lens_Iris_DB"):
    x_train, y_train = [], []
    x_test, y_test = [], []

    # Primeiro carrego as imagens de treino
    with open(train_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            img_path = row[0] # Path da imagem
            label    = row[1] # Label da imagem
            x_train.append(list(resize(imread(os.path.join(base_path, img_path)), dsize=(300, 300))))
            y_train.append(one_hot_encode_label(label))
    
    # Depois as de teste
    with open(test_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            img_path = row[0] # Path da imagem
            label    = row[1] # Label da imagem
            x_test.append(list(resize(imread(os.path.join(base_path, img_path)), dsize=(300, 300))))
            y_test.append(one_hot_encode_label(label))
    
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

if __name__ == "__main__":
    # Carregando o dataset
    x_train, x_test, y_train, y_test = lens_dataset_csv()

    """
    neuralNet = tf.keras.models.load_model("./contact-lens-model-multitask_v3_b0")
    neuralNet = tf.keras.Model(inputs=neuralNet.input, outputs=neuralNet.get_layer("dense_3").output)

    print("Transfer learning base: ")
    neuralNet.summary()

    neuralNet = tf.keras.layers.Dense(128, activation='relu')(neuralNet)
    neuralNet = tf.keras.layers.Dense(64, activation='relu')(neuralNet)
    neuralNet = tf.keras.layers.Dense(3, activation="softmax")(neuralNet)  # Output

    """
    #inputs = tf.keras.layers.Input(shape=(300,300,3))
    neuralNet = tf.keras.models.load_model("./contact-lens-model-multitask_v3_b0", compile=False)
    neuralNet.trainable = False # O modelo base nao treina
    pre_treinada_saida = tf.keras.layers.Dense(128, activation='relu', name="Grande")(neuralNet.get_layer("dense_3").output)
    pre_treinada_saida = tf.keras.layers.Dense(64, activation='relu', name="Camada")(pre_treinada_saida)
    out = tf.keras.layers.Dense(3, activation="softmax", name="Bacana")(pre_treinada_saida)  # Output

    # Construção do modelo
    neuralNet = tf.keras.Model(neuralNet.input, out)

    print("\n\n\n\nWhole model: ")
    neuralNet.summary()

    neuralNet.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
    
    neuralNet.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test), verbose=1)
    neuralNet.save("./contact-lens-model-with_transfer_learning_v2")
