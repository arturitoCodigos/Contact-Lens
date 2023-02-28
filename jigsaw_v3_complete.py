import os
import numpy as np
import tensorflow as tf

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
        return [1,0,0,0,0,0]
    if lbl == three_way_partition[1]:
        return [0,1,0,0,0,0]
    if lbl == three_way_partition[2]:
        return [0,0,1,0,0,0]
    if lbl == three_way_partition[3]:
        return [0,0,0,1,0,0]
    if lbl == three_way_partition[4]:
        return [0,0,0,0,1,0]
    if lbl == three_way_partition[5]:
        return [0,0,0,0,0,1]

def readDir(path, info=True):
    labels = []
    files = os.listdir(path)
    if info:
        print(f"Sao {len(files)} arquivos a serem lidos!")
        i = 0
    for img_name in files:
        img_name_ = os.path.splitext(img_name)[0]
        try:
            labels.append(oneHot(list(map(toInt, img_name_.split("=")[1].split("-")))))
        except:
            print(f"Erro no arquivo {img_name}, removendo-o...")
            os.remove(os.path.join(path, img_name))
            continue
        if info:
            if (i%100 == 0):
                print(f"Estamos no arquivo numero {i}!")
            i+=1
    return labels

if __name__ == "__main__":
    # Carregando o dataset
    y_branch1, y_branch2, y_branch3 = readDir("./three_way_dataset/dir_001"), readDir("./three_way_dataset/dir_002"), readDir("./three_way_dataset/dir_003")
    labels = y_branch1 + y_branch2 + y_branch3

    train = tf.keras.preprocessing.image_dataset_from_directory("./three_way_dataset", labels=labels, image_size=(300, 300), shuffle=True, seed=55, validation_split=0.2, subset="training")
    test = tf.keras.preprocessing.image_dataset_from_directory("./three_way_dataset", labels=labels, image_size=(300, 300), shuffle=True, seed=55, validation_split=0.2, subset="validation")

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
    main_branch = tf.keras.layers.Dense(6, activation='softmax')(main_branch)

    # Construção do modelo
    neuralNet = tf.keras.Model(inputs, outputs=main_branch)
                                                     
    neuralNet.summary() # Resumo da rede

    neuralNet.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])

    neuralNet.fit(train,
                  epochs=40,
                  verbose=1)
    
    print("\n\n\n\n EVALUATION NOW: \n\n\n\n")

    neuralNet.evaluate(test, verbose=1)
    
    neuralNet.save("./contact-lens-model-_v3_b0_FULLIMGS")
