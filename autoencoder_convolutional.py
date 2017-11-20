from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import TensorBoard
import numpy as np
import config


def train_autoencoder():
    # get parameters
    width = config.width
    height = config.height
    if config.grayscale == 1:
        color_layers = 1
    else:
        color_layers = 3
    train_size = config.train_size
    test_size = config.test_size
    epochs = config.epochs
    batch_size = config.batch_size

    # define autoencoder model
    input_img = Input(shape=(height, width, color_layers))

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # get train and test data
    x_train = np.zeros(shape=(train_size, height, width, color_layers))
    for i in range(0, train_size):
        img = load_img('processed_data/' + str(i + 1) + '.jpg')
        x = img_to_array(img)
        x_train[i] = x

    x_test = np.zeros(shape=(test_size, height, width, color_layers))
    for i in range(train_size, train_size + test_size):
        img = load_img('processed_data/' + str(i + 1) + '.jpg')
        x = img_to_array(img)
        x_test[i - train_size] = x

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    print('Train shape:')
    print(x_train.shape)
    print('Test shape')
    print(x_test.shape)

    # train model
    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # save encoder model
    encoder.save("encoder.h5")
