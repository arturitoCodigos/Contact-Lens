import os
import numpy as np
import tensorflow as tf
import random
import cv2


class ImagenetDataLoader(tf.keras.utils.Sequence):
    """
    Imagenet data loader.
    Based on: https://stackoverflow.com/questions/49510612/change-training-dataset-every-n-epochs-in-keras.
    """

    def __init__(self, x_set, y_set, load_input_shape, net_input_shape, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.load_input_shape = load_input_shape[0:2]
        self.net_input_shape = net_input_shape[0:2]
        self.diff_input_shape = (load_input_shape[0] - net_input_shape[0] + 1, load_input_shape[1] - net_input_shape[1] + 1)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_data_x = np.zeros(shape=(self.batch_size, self.net_input_shape[0], self.net_input_shape[1], 3))
        for e, file_name in enumerate(batch_x):
            batch_data_x[e] = self._data_load(file_name)
        # return np.array([self._data_load(file_name) for file_name in batch_x]), batch_y
        return batch_data_x, batch_y

    def _data_load(self, file_name):
        img = cv2.resize(cv2.imread(file_name, cv2.IMREAD_COLOR), self.load_input_shape, interpolation=cv2.INTER_AREA)
        x_begin = random.randrange(self.diff_input_shape[0])
        y_begin = random.randrange(self.diff_input_shape[1])
        return img[x_begin:x_begin + self.net_input_shape[0], y_begin:y_begin + self.net_input_shape[1], :] / 255.


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
        return 0
    if lbl == three_way_partition[1]:
        return 1
    if lbl == three_way_partition[2]:
        return 2
    if lbl == three_way_partition[3]:
        return 3
    if lbl == three_way_partition[4]:
        return 4
    if lbl == three_way_partition[5]:
        return 5

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

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

if __name__ == "__main__":

    # Nomes dos arquivos
    _x_train = listdir_fullpath("./three_way_dataset/dir_001") + listdir_fullpath("./three_way_dataset/dir_002") + listdir_fullpath("./three_way_dataset/dir_003")

    # Labels
    y_branch1, y_branch2 = readDir("./three_way_dataset/dir_001"), readDir("./three_way_dataset/dir_002")
    _y_train = y_branch1 + y_branch2

    # Val set
    _x_test = listdir_fullpath("./three_way_dataset/dir_003")
    _y_test = readDir("./three_way_dataset/dir_003")


    train_data_gen = ImagenetDataLoader(x_set=_x_train,
                                        y_set=tf.keras.utils.to_categorical(_y_train, num_classes=6),
                                        load_input_shape=(300,300,3),
                                        net_input_shape=(300,300,3),
                                        batch_size=32)


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

    neuralNet.fit(train_data_gen,
                  shuffle=True,
                  epochs=40,
                  validation_data=(_x_test, tf.keras.utils.to_categorical(_y_test, num_classes=6)),
                  verbose=1)
    
    #print("\n\n\n\n EVALUATION NOW: \n\n\n\n")

    #neuralNet.evaluate(test, verbose=1)
    
    neuralNet.save("./contact-lens-model-_v4_b0_FULLIMGS")
