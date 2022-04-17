from tensorflow import keras
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy


TRAIN_FILE = 'data/train_data.txt'
TEST_FILE = 'data/test_data.txt'
NUMBER_COUNT = 10


class Data:
    def __init__(self, inputs, result=None):
        self.inputs = inputs
        self.results = [-1] * NUMBER_COUNT
        if result is not None:
            self.results[result] = 1


def convert_data9(data: List[List[List[float]]]) -> List[str]:
    """
    Compresses image in 9 times

    :param data: array of images
    :return: List of strings in 0/1 format
    """
    imgArr = []
    for image in data:
        temp = ''
        for j in range(0, len(image) - 2, 3):
            for k in range(0, len(image[0]) - 2, 3):
                s = image[j][k] + image[j + 1][k] + image[j + 2][k] + \
                    image[j][k + 1] + image[j + 1][k + 1] + image[j + 2][k + 1] + \
                    image[j][k + 2] + image[j + 1][k + 2] + image[j + 2][k + 2]
                s = '1' if s > 0.5 * 9 else '0'
                temp += s
        imgArr.append(temp)
    return imgArr


def convert_data4(data: List[List[List[float]]]) -> List[str]:
    """
    Compresses image in 4 times

    :param data: array of images
    :return: List of strings in 0/1 format
    """
    imgArr = []
    for image in data:
        temp = ''
        for j in range(0, len(image) - 1, 2):
            for k in range(0, len(image[0]) - 1, 2):
                s = image[j][k] + image[j + 1][k] + \
                    image[j][k + 1] + image[j + 1][k + 1]
                s = '1' if s > 0.5 * 4 else '0'
                temp += s
        imgArr.append(temp)
    return imgArr


def draw(data: List[List[List[float]]]) -> None:
    """
    Draws first 30 images

    :param data: Array of images
    :return: Picture
    """
    plt.figure(figsize=(10, 3))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(data[i], cmap='gray')
        plt.axis('off')
    plt.show()


def download_data() -> Tuple[numpy.ndarray, List[int],
                             numpy.ndarray, List[int]]:
    """
    Downloads MNIST data of images.

    :return: training and testing dataset
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


def get(filename: str) -> Dict[str, int]:
    f = open(filename)

    data = {}
    for lines in f:
        strings = lines.split(':')

        key = strings[0]
        value = int(strings[1].rstrip())

        data[key] = value

    return data


def save(filename: str, data: Dict[str, int]) -> None:
    """
    Saves reformat images in file

    :param filename: name of file
    :param data: initial dataset in format of dictionary
    :return: None
    """
    f = open(filename, 'w')

    for item, value in data.items():
        s = item + ':' + str(value) + '\n'
        f.write(s)
    f.close()


def create_input(data: List[str], values: List[int]) -> Dict[str, int]:
    res = {}
    for image, value in zip(data, values):
        res[image] = value
    return res


def init_data():
    """
    Inits datasets

    :return: train and test datasets
    """
    x_train, y_train, x_test, y_test = download_data()
    x_train = x_train / 255
    x_test = x_test / 255

    draw(x_train)

    x_train_str = convert_data4(x_train)
    x_test_str = convert_data4(x_test)

    train_dict = create_input(x_train_str, y_train)
    test_dict = create_input(x_test_str, y_test)

    save(TRAIN_FILE, train_dict)
    save(TEST_FILE, test_dict)


def get_dataset(filename) -> List[Data]:
    """
    Gets dataset from file

    :param filename: name of file
    :return: dataset
    """
    values = get(filename)

    dataset = [Data([int(ch) for ch in string], result)
            for string, result in values.items()]

    print(">>> Dataset is ready")
    return dataset


def main():
    init_data()


if __name__ == '__main__':
    main()
