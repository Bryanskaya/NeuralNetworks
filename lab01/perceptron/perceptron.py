from perceptron.layers import *
from typing import List
from training_dataset import Data

import numpy as np
import math
import random


class Perceptron:
    def __init__(self):
        self.s_layer = SLayer()
        self.a_layer = ALayer()
        self.r_layer = RLayer()

    def solve(self, inputs: List[int]) -> List[List[int]]:
        s_result = self.s_layer.solve(inputs)
        a_result = self.a_layer.solve(s_result)
        return self.r_layer.solve(a_result)

    def correct(self, expected_results: List[int]) -> None:
        self.r_layer.correct(expected_results)

    def train(self, dataset: List[Data]) -> None:
        """
        Training process

        :param dataset: List[Data]
        :return: None
        """
        print('>>> Training start')

        continue_training = True
        epoch = 0

        total_classifications = len(dataset) * len(dataset[0].results)
        min_wrong_classifications = total_classifications
        stability_time = 0
        while continue_training and stability_time < 100:
            wrong_classifications = 0
            continue_training = False

            random.shuffle(dataset)
            for data in dataset:
                results = self.solve(data.inputs)

                for result, expected_result in zip(results, data.results):
                    if result != expected_result:
                        wrong_classifications += 1

                        self.correct(data.results)

                        continue_training = True
                        break

            epoch += 1
            if epoch % 1 == 0:
                print('Epoch {:d} ended. Wrong classifications: {:d}'.format(epoch, wrong_classifications))

            if min_wrong_classifications <= wrong_classifications:
                stability_time += 1
            else:
                min_wrong_classifications = wrong_classifications
                stability_time = 0

        print('Training ended in {:d} epochs\nResult accurancy on training dataset: {:.1f}%'.format(
                epoch,
                float(total_classifications - min_wrong_classifications) / total_classifications * 100))

    def optimize(self, dataset: List[Data]) -> None:
        """
        Delete dead or correlating neurons

        :param dataset: List[Data]
        :return: None
        """
        print('----------------------------')
        print('>>> Starting optimization')

        '''
        for data in dataset:
            self.a_layer.solve(data.inputs)
        for neuron in self.a_layer.neurons:
            neuron.bias = (neuron.max_acc - neuron.min_acc) * 0.7 + neuron.min_acc
        '''

        activations = []
        for _ in self.a_layer.neurons:
            activations.append([])
        a_inputs = [self.s_layer.solve(data.inputs) for data in dataset]
        for i_count, a_input in enumerate(a_inputs):
            for n_count, neuron in enumerate(self.a_layer.neurons):
                activations[n_count].append(neuron.solve(a_input))
        to_remove = [False] * len(self.a_layer.neurons)

        print('Dead neurons from A-layer        =', end=' ')
        for i, activation in enumerate(activations):
            zeros = activation.count(0)
            if zeros == 0 or zeros == len(a_inputs):
                to_remove[i] = True
        dead_neurons = to_remove.count(True)
        print('{:d}'.format(dead_neurons))

        print('Correlating neurons from A-layer =', end=' ')
        for i in range(len(activations) - 1):
            if to_remove[i]:
                continue

            for j in range(i + 1, len(activations)):
                if to_remove[j]:
                    continue
                #if activations[i] == activations[j]:
                if math.fabs(self.kPirson(activations[i], activations[j])) > 0.8:
                    to_remove[j] = True
        print('{:d}'.format(to_remove.count(True) - dead_neurons))

        for i in range(len(to_remove) - 1, -1, -1):
            if to_remove[i]:
                del self.a_layer.neurons[i]
                for j in range(len(self.r_layer.neurons)):
                    del self.r_layer.neurons[j].input_weights[i]

        print('Neurons remaining = {:18d}'.format(len(self.a_layer.neurons)))

    def kPirson(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        v1 = v1 - np.mean(v1)
        v2 = v2 - np.mean(v2)

        sum1 = np.sum(v1 * v2)
        sum21 = np.sum(v1 ** 2)
        sum22 = np.sum(v2 ** 2)

        return sum1 / math.sqrt(sum21 * sum22)

    def getS(self, v):
        m = np.mean(v)
        temp = 0
        for elem in v:
            temp += (elem - m) ** 2
        return temp