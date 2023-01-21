import os
import numpy as np
import tensorflow as tf
from cv2 import imread
from sklearn.model_selection import train_test_split

def oneHot(num):
    return [1 if int(num) == (i+1) else 0 for i in range(9)]

def readDir(path="./imgs", info=True):
    features, labels = [], []
    files = os.listdir(path)
    if info:
        print(f"Sao {len(files)} arquivos a serem lidos!")
        i = 0
    for img_name in files:
        features.append(imread(os.path.join(path, img_name)))
        img_name = os.path.splitext(img_name)[0]
        labels.append(list(map(oneHot, img_name.split("-"))))
        if info:
            if (i%100 == 0):
                print(f"Estamos no arquivo numero {i}!")
            i+=1
    return np.array(features), np.array(labels)

def divideLabels(labels, info=True):
    o1, o2, o3, o4, o5, o6, o7, o8, o9 = [], [], [], [], [], [], [], [], []
    if info:
        print(f"\n\n\nSao {len(labels)} labels a serem subdividas!")
        i = 0
    for label in labels: # Cada `label` eh um array contendo os one-hot encoded vectors
        o1.append(label[0])
        o2.append(label[1])
        o3.append(label[2])
        o4.append(label[3])
        o5.append(label[4])
        o6.append(label[5])
        o7.append(label[6])
        o8.append(label[7])
        o9.append(label[8])
        if info:
            if (i%100 == 0):
                print(f"Estamos no arquivo numero {i}!")
            i+=1
    return np.array(o1), np.array(o2), np.array(o3), np.array(o4), np.array(o5), np.array(o6), np.array(o7), np.array(o8), np.array(o9)


def create_branch(base_model, name):
    return tf.keras.layers.Dense(9, activation='softmax', name=name)(base_model)

if __name__ == "__main__":
    # Carregando o dataset
    features, labels = readDir()

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=23, shuffle=True)
    y_train_branch1, y_train_branch2, y_train_branch3, y_train_branch4, y_train_branch5, y_train_branch6, y_train_branch7, y_train_branch8, y_train_branch9 = divideLabels(y_train)
    y_test_branch1, y_test_branch2, y_test_branch3, y_test_branch4, y_test_branch5, y_test_branch6, y_test_branch7, y_test_branch8, y_test_branch9 = divideLabels(y_test)
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
    out4 = create_branch(main_branch, 'out4')
    out5 = create_branch(main_branch, 'out5')
    out6 = create_branch(main_branch, 'out6')
    out7 = create_branch(main_branch, 'out7')
    out8 = create_branch(main_branch, 'out8')
    out9 = create_branch(main_branch, 'out9')

    # Construção do modelo
    neuralNet = tf.keras.Model(inputs, outputs=[out1, out2, out3, 
                                                out4, out5, out6,
                                                out7, out8, out9])
                                                     

    #neuralNet.summary() # Resumo da rede

    neuralNet.compile(loss={
                            'out1' : 'categorical_crossentropy', 
                            'out2' : 'categorical_crossentropy',
                            'out3' : 'categorical_crossentropy',
                            'out4' : 'categorical_crossentropy',
                            'out5' : 'categorical_crossentropy',
                            'out6' : 'categorical_crossentropy',
                            'out7' : 'categorical_crossentropy',
                            'out8' : 'categorical_crossentropy',
                            'out9' : 'categorical_crossentropy',
                        },
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
    neuralNet.fit(x_train,
                  {
                    'out1' : y_train_branch1, 
                    'out2' : y_train_branch2,
                    'out3' : y_train_branch3,
                    'out4' : y_train_branch4,
                    'out5' : y_train_branch5,
                    'out6' : y_train_branch6,
                    'out7' : y_train_branch7,
                    'out8' : y_train_branch8,
                    'out9' : y_train_branch9,
                  }, 
                  epochs=40, 
                  validation_data=(x_test, {
                    'out1' : y_test_branch1, 
                    'out2' : y_test_branch2,
                    'out3' : y_test_branch3,
                    'out4' : y_test_branch4,
                    'out5' : y_test_branch5,
                    'out6' : y_test_branch6,
                    'out7' : y_test_branch7,
                    'out8' : y_test_branch8,
                    'out9' : y_test_branch9,
                  }), 
                  verbose=1)
    neuralNet.save("./contact-lens-model-multitask_b0")
