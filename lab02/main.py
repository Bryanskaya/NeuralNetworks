import struct
import numpy as np

from funcs import *

path = 'MNIST'


class Perceptron(object):
    batch = 4000
    neuronH = 100
    eta = 0.1
    epoch = 400
    tanhParams = None

    def __init__(self):
        self.fobj = TanFunc()
        self.x_train = self.img_process(self.read_idx(path + '/train-images.idx3-ubyte'))
        self.y_train = self.encoding(self.read_idx(path + '/train-labels-idx1-ubyte'))
        self.x_test = self.img_process(self.read_idx(path + '/t10k-images-idx3-ubyte'))
        self.y_test = self.read_idx(path + '/t10k-labels-idx1-ubyte')

        self.neuronS = self.x_train.shape[0]

        self.init_f_activation()

    def img_process(self, data):
        data = data / 255
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        return data.T

    def read_idx(self, filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    def encoding(self, label):
        n = np.max(label) + 1
        v = np.eye(n)[label]
        return v.T

    def init_f_activation(self):
        self.tanhParams = {'W1': np.random.randn(self.neuronH, self.neuronS) * np.sqrt(1. / self.neuronS),  # matr
                           'b1': np.zeros((self.neuronH, 1)),
                           'W2': np.random.randn(10, self.neuronH) * np.sqrt(1. / self.neuronH),
                           'b2': np.zeros((10, 1))
                           }

    def train(self):
        for i in range(self.epoch):
            idx = np.random.permutation(self.x_train.shape[1])[:self.batch]  # перемешать
            x = self.x_train[:, idx]
            y = self.y_train[:, idx]

            forwardPass = self.forward(x, self.tanhParams, self.fobj.f)
            gradient = self.back(x, y, forwardPass, self.tanhParams, self.fobj.df)
            self.tanhParams = self.updater(x, self.tanhParams, gradient, self.eta)  # обновить веса
            if i % 10 == 0:
                acc_test = self.calc_accuracy(self.x_test, self.y_test, self.tanhParams)
                print(f"epoch {i} was finished, accuracy\t{acc_test * 100:.2f}%")

    def forward(self, x, params, activation):
        forwardPass = {}
        forwardPass['Z1'] = np.matmul(params['W1'], x) + params['b1']
        forwardPass['A1'] = activation(forwardPass['Z1'])
        forwardPass['Z2'] = np.matmul(params['W2'], forwardPass['A1']) + params['b2']
        forwardPass['A2'] = self.softMax(forwardPass['Z2'])
        return forwardPass

    def softMax(self, x):
        e = np.exp(x)
        return e / np.sum(e, axis=0)

    def back(self, x, y, forwardPass, params, dActivation):
        gradient = {}
        gradient['dZ2'] = forwardPass['A2'] - y
        gradient['dW2'] = np.matmul(gradient['dZ2'], forwardPass['A1'].T)
        gradient['db2'] = np.sum(gradient['dZ2'], axis=1, keepdims=True)
        gradient['dA1'] = np.matmul(params['W2'].T, gradient['dZ2'])
        # print(gradient['dA1'])
        gradient['dZ1'] = gradient['dA1'] * dActivation(forwardPass['A1'])
        gradient['dW1'] = np.matmul(gradient['dZ1'], x.T)
        gradient['db1'] = np.sum(gradient['dZ1'])
        return gradient

    def updater(self, x, params, grad, eta):
        updatedParams = {}
        m = x.shape[1]

        updatedParams['W2'] = params['W2'] - eta * grad['dW2'] * (1. / m)
        updatedParams['b2'] = params['b2'] - eta * grad['db2'] * (1. / m)
        updatedParams['W1'] = params['W1'] - eta * grad['dW1'] * (1. / m)
        updatedParams['b1'] = params['b1'] - eta * grad['db1'] * (1. / m)
        return updatedParams

    def classifer(self, x, y, params, activation):
        forwardPass = self.forward(x, params, activation)

        pred = np.argmax(forwardPass['A2'], axis=0)
        gradient = self.back(x, y, forwardPass, params, self.fobj.df)
        print(f"Average error on hidden layer\t {np.abs(np.mean(gradient['dZ1'])) * 100 :.2f}%")
        print(gradient['dA1'])
        return pred

    def calc_accuracy(self, x_test, y_test, params):
        y_hat = self.classifer(x_test, y_test, params, self.fobj.f)
        return sum(y_hat == y_test) * 1 / len(y_test)

    def get_accuracy(self):
        acc = self.calc_accuracy(self.x_test, self.y_test, self.tanhParams)
        print('Total accuracy {:.1f}%'.format(acc * 100))


def main():
    np.random.seed(7)

    perceptron = Perceptron()
    perceptron.train()
    perceptron.get_accuracy()


if __name__ == '__main__':
    main()
