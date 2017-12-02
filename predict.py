import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import config
import cv2


def clusterization():
    # get config values
    if config.grayscale == 1:
        color_layers = 1
    else:
        color_layers = 3

    # load model
    model = load_model("encoder.h5")

    # get data
    # get train and test data
    x_train = get_train_data(color_layers)

    # get encoded representation of data
    encoded_img = model.predict(x_train)
    print('Encoded image shape')
    print(encoded_img.shape)
    encoded_img = encoded_img.reshape(len(encoded_img), encoded_img.shape[1] * encoded_img.shape[2] * encoded_img.shape[3])
    print('Reshaped to')
    print(encoded_img.shape)

    print('Running kmeans on encoded images')
    if config.kmeansDistance == 1:
        sklearn_euclidean_kmeans(encoded_img)
    elif config.kmeansDistance == 2:
        nltk_cosine_kmeans(encoded_img)
    elif config.kmeansDistance == 3:
        nltk_euclidean_kmeans(encoded_img)
    elif config.kmeansDistance == 4:
        nltk_manhattan_kmeans(encoded_img)


def sklearn_euclidean_kmeans(encoded_img):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(encoded_img)

    print_labels(kmeans.labels_)


def nltk_cosine_kmeans(encoded_img):
    from nltk.cluster.util import cosine_distance
    from nltk.cluster.kmeans import KMeansClusterer

    kclusterer = KMeansClusterer(2, distance=cosine_distance, repeats=10)
    assigned_clusters = kclusterer.cluster(encoded_img, assign_clusters=True)

    print_labels(assigned_clusters)


def nltk_euclidean_kmeans(encoded_img):
    from nltk.cluster.util import euclidean_distance
    from nltk.cluster.kmeans import KMeansClusterer

    kclusterer = KMeansClusterer(2, distance=euclidean_distance, repeats=10)
    assigned_clusters = kclusterer.cluster(encoded_img, assign_clusters=True)

    print_labels(assigned_clusters)


def nltk_manhattan_kmeans(encoded_img):
    from scipy.spatial.distance import cityblock
    from nltk.cluster.kmeans import KMeansClusterer

    kclusterer = KMeansClusterer(2, distance=cityblock, repeats=10)
    assigned_clusters = kclusterer.cluster(encoded_img, assign_clusters=True)

    print_labels(assigned_clusters)


def print_labels(labels):
    print('Saving kmeans results to file')
    thefile = open('unsupervised.txt', 'w')
    for item in labels:
        thefile.write("%s\n" % item)


def get_train_data(color_layers):
    if color_layers == 1:
        x_train = np.zeros(shape=(config.data_size, config.height, config.width))
        for i in range(0, config.data_size):
            x = cv2.imread("processed_data/" + str(i + 1) + ".jpg", cv2.IMREAD_GRAYSCALE)
            x_train[i] = x
    else:
        x_train = np.zeros(shape=(config.data_size, config.height, config.width, color_layers))
        for i in range(0, config.data_size):
            img = load_img('processed_data/' + str(i + 1) + '.jpg')
            x = img_to_array(img)
            x_train[i] = x

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), config.height, config.width, color_layers))
    return x_train
