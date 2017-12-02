data_size = 1000
train_size = 900
test_size = 100
height = 200
width = 200
grayscale = 1  # 1 for grayscale conversion
epochs = 10
batch_size = 100

# set of images containing clear bushes of hydrangea
# used to determine the labels set by kmeans
reference_set = [3, 7, 27, 44, 47, 52, 56]

# distance metrics for kmeans
# 1 - euclidean, sklearn library
# 2 - cosine, nltk library
# 3 - euclidean, nltk library
# 4 - manhattan, nltk library
kmeansDistance = 4

