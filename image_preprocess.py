import cv2
import config


def process():
    for i in range(1, config.data_size + 1):
        im = cv2.imread("data/" + str(i) + ".jpg")
        if config.grayscale == 1:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = cv2.resize(im, (config.width, config.height))
        cv2.imwrite("processed_data/" + str(i) + ".jpg", im)