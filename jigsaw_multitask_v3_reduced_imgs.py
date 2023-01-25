import os
import numpy as np
import tensorflow as tf
from cv2 import imread
from sklearn.model_selection import train_test_split

"""
1 2 3
1 3 2
2 1 3
2 3 1
3 1 2
3 2 1
"""

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

def toInt(s):
    return int(s)

def oneHot(lbl):
    if lbl == three_way_partition[0]:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if lbl == three_way_partition[1]:
        return [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    if lbl == three_way_partition[2]:
        return [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    if lbl == three_way_partition[3]:
        return [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    if lbl == three_way_partition[4]:
        return [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    if lbl == three_way_partition[5]:
        return [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

def readDir(path="./three_way_dataset", info=True, max=5000):
    features, labels = [], []
    files = os.listdir(path)
    limiter = 0
    if info:
        print(f"Sao {len(files)} arquivos a serem lidos!")
        i = 0
    for img_name in files:
        limiter += 1
        if (limiter >= max):
            break
        features.append(imread(os.path.join(path, img_name)))
        img_name = os.path.splitext(img_name)[0]
        try:
            labels.append(oneHot(list(map(toInt, img_name.split("=")[1].split("-")))))
        except:
            print(f"Erro no arquivo {img_name}")
            features.pop()
            continue
        if info:
            if (i%100 == 0):
                print(f"Estamos no arquivo numero {i}!")
            i+=1
    return np.array(features, dtype=object), np.array(labels, dtype=object)

def divideLabels(labels, info=True):
    o1, o2, o3 = [], [], []
    if info:
        print(f"\n\n\nSao {len(labels)} labels a serem subdividas!")
        i = 0
    for label in labels: # Cada `label` eh um array contendo os one-hot encoded vectors
        o1.append(label[0])
        o2.append(label[1])
        o3.append(label[2])
        if info:
            if (i%100 == 0):
                print(f"Estamos no arquivo numero {i}!")
            i+=1
    return np.array(o1), np.array(o2), np.array(o3)


def create_branch(base_model, name):
    return tf.keras.layers.Dense(3, activation='softmax', name=name)(base_model)

if __name__ == "__main__":
    # Carregando o dataset
    features, labels = readDir()

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=23, shuffle=True)
    y_train_branch1, y_train_branch2, y_train_branch3 = divideLabels(y_train)
    y_test_branch1, y_test_branch2, y_test_branch3    = divideLabels(y_test)
    # Neural Net em si

    # Modelo 1 --> Soprado da documentação do keras

    inputs = tf.keras.layers.Input(shape=(300,300,3))
    pre_treinada_saida = inputs

    main_branch = tf.keras.applications.EfficientNetB0(weights="imagenet", input_tensor=pre_treinada_saida, include_top=False)
    main_branch.trainable = True # O modelo base treina?
    main_branch = tf.keras.layers.GlobalAveragePooling2D()(main_branch.output)
    main_branch = tf.keras.layers.BatchNormalization()(main_branch)
    main_branch = tf.keras.layers.Dense(1024, activation='relu')(main_branch)
    main_branch = tf.keras.layers.Dense(512, activation='relu')(main_branch)
    main_branch = tf.keras.layers.Dense(256, activation='relu')(main_branch)
    main_branch = tf.keras.layers.Dense(128, activation='relu')(main_branch)

    out1 = create_branch(main_branch, 'out1')
    out2 = create_branch(main_branch, 'out2')
    out3 = create_branch(main_branch, 'out3')

    # Construção do modelo
    neuralNet = tf.keras.Model(inputs, outputs=[out1, out2, out3])
                                                     
    neuralNet.summary() # Resumo da rede

    neuralNet.compile(loss={
                            'out1' : 'categorical_crossentropy', 
                            'out2' : 'categorical_crossentropy',
                            'out3' : 'categorical_crossentropy',
                        },
                        loss_weights={
                            'out1' : 0.33,
                            'out2' : 0.33, 
                            'out3' : 0.33,
                        },
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
    neuralNet.fit(x_train,
                  {
                    'out1' : y_train_branch1, 
                    'out2' : y_train_branch2,
                    'out3' : y_train_branch3,
                  }, 
                  epochs=40, 
                  validation_data=(x_test, {
                    'out1' : y_test_branch1, 
                    'out2' : y_test_branch2,
                    'out3' : y_test_branch3,
                  }), 
                  verbose=1)
    neuralNet.save("./contact-lens-model-multitask_v3_b0")
