from perceptron.layers import *
from typing import List
from training_dataset import Data

import random


class Perceptron:
    def __init__(self):
        self.s_layer = SNeuronLayer()
        self.a_layer = ANeuronLayer()
        self.r_layer = RNeuronLayer()

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

        activations = []
        for _ in self.a_layer.neurons:
            activations.append([])
        a_inputs = [self.s_layer.solve(data.inputs) for data in dataset]
        for i_count, a_input in enumerate(a_inputs):
            for n_count, neuron in enumerate(self.a_layer.neurons):
                activations[n_count].append(neuron.solve(a_input))
        to_remove = [False] * len(self.a_layer.neurons)

        a_layer_size = len(self.a_layer.neurons)

        print('Dead neurons from A-layer        =', end=' ')
        for i, activation in enumerate(activations):
            zeros = activation.count(0)
            if zeros == 0 or zeros == a_layer_size:
                to_remove[i] = True
        dead_neurons = to_remove.count(True)
        print('{:d}'.format(dead_neurons))

        print('Correlating neurons from A-layer =', end=' ')
        for i in range(len(activations) - 1):
            if not to_remove[i]:
                for j in range(i + 1, len(activations)):
                    if activations[j] == activations[i]:
                        to_remove[j] = True
        correlating_neurons = to_remove.count(True) - dead_neurons
        print('{:d}'.format(correlating_neurons))

        for i in range(len(to_remove) - 1, -1, -1):
            if to_remove[i]:
                del self.a_layer.neurons[i]
                for j in range(len(self.r_layer.neurons)):
                    del self.r_layer.neurons[j].input_weights[i]

        print('Neurons remaining = {:18d}'.format(len(self.a_layer.neurons)))
