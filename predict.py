from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import config
from sklearn.cluster import KMeans


def clusterization():
    # get config values
    height = config.height
    width = config.width
    data_size = config.data_size
    if config.grayscale == 1:
        color_layers = 1
    else:
        color_layers = 3

    # load model
    model = load_model("encoder.h5")

    # get data
    x_train = np.zeros(shape=(data_size, height, width, color_layers))
    for i in range(0, data_size):
        img = load_img('processed_data/' + str(i + 1) + '.jpg')
        x = img_to_array(img)
        x_train[i] = x
    x_train = x_train.astype('float32') / 255.

    # get encoded representation of data
    encoded_img = model.predict(x_train)
    print('Encoded image shape')
    print(encoded_img.shape)
    encoded_img = encoded_img.reshape(len(encoded_img), encoded_img.shape[1] * encoded_img.shape[2] * encoded_img.shape[3])
    print('Reshaped to')
    print(encoded_img.shape)

    print('Running kmeans on encoded images')
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(encoded_img)

    print('Saving kmeans results to file')
    thefile = open('unsupervised.txt', 'w')
    for item in kmeans.labels_:
        thefile.write("%s\n" % item)
