import cv2
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img

import config


def train_autoencoder():
    # get parameters
    width = config.width
    height = config.height
    if config.grayscale == 1:
        color_layers = 1
    else:
        color_layers = 3
    epochs = config.epochs
    batch_size = config.batch_size

    # define autoencoder model
    input_img = Input(shape=(height, width, color_layers))
    print("Input shape")
    print(input_img)

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(color_layers, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train = get_train_data(color_layers)
    x_test = get_test_data(color_layers)

    # train model
    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # save encoder model
    encoder.save("encoder.h5")


def get_train_data(color_layers):
    if color_layers == 1:
        x_train = np.zeros(shape=(config.train_size, config.height, config.width))
        for i in range(0, config.train_size):
            x = cv2.imread("processed_data/" + str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)
            x_train[i] = x
    else:
        x_train = np.zeros(shape=(config.train_size, config.height, config.width, color_layers))
        for i in range(0, config.train_size):
            img = load_img('processed_data/' + str(i + 1) + '.jpg')
            x = img_to_array(img)
            x_train[i] = x

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), config.height, config.width, color_layers))
    print('Train shape:')
    print(x_train.shape)
    return x_train


def get_test_data(color_layers):
    if color_layers == 1:
        x_test = np.zeros(shape=(config.test_size, config.height, config.width))
        for i in range(config.train_size, config.train_size + config.test_size):
            x = cv2.imread("processed_data/" + str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)
            x_test[i - config.train_size] = x
    else:
        x_test = np.zeros(shape=(config.test_size, config.height, config.width, color_layers))
        for i in range(config.train_size, config.train_size + config.test_size):
            img = load_img('processed_data/' + str(i + 1) + '.jpg')
            x = img_to_array(img)
            x_test[i - config.train_size] = x

    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), config.height, config.width, color_layers))
    print('Test shape')
    print(x_test.shape)
    return x_test

