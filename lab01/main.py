from perceptron.perceptron import Perceptron
from perceptron.layers import *
from training_dataset import get_dataset, Data, NUMBER_COUNT, TRAIN_FILE, TEST_FILE
import random


def init_perceptron(dataset: List[Data]) -> Perceptron:
    """
    Inits perceptron (s/a/r-layers).

    :param dataset: training dataset
    :return: Perceptron
    """
    network = Perceptron()
    input_count = len(dataset[0].inputs)

    for _ in range(input_count):
        network.s_layer.add_neuron(None, lambda value: value)

    #a_neurons_count = 2 ** input_count - 1
    a_neurons_count = 700
    for position in range(a_neurons_count):
        neuron = ANeuron(None, lambda value: int(value >= 0))
        neuron.input_weights = [random.choice([-1, 0, 1]) for _ in range(input_count)]
        neuron.calculate_bias()
        network.a_layer.neurons.append(neuron)

    for _ in range(NUMBER_COUNT):
        network.r_layer.add_neuron(a_neurons_count, lambda: 0, lambda value: 1 if value >= 0 else -1, 0.01, 0)

    return network


def train_perceptron(network: Perceptron, dataset: List[Data]) -> Perceptron:
    """
    Trains and optimizes perceptron

    :param network: Perceptron
    :param dataset: training dataset
    :return: modified perceptron
    """
    network.train(dataset)
    network.optimize(dataset)

    return network


def test_network(network: Perceptron, test_dataset: List[Data]) -> None:
    """
    Tests perceptron

    :param network: Perceptron
    :param test_dataset: testing dataset
    :return: None
    """
    total_classifications = len(test_dataset) * len(test_dataset[0].results)
    misc = 0
    for data in test_dataset:
        results = network.solve(data.inputs)
        for result, expected_result in zip(results, data.results):
            if result != expected_result:
                misc += 1

    print('----------------------------')
    print('Test accuracy: {:.2f}%'.format(float(total_classifications - misc) / total_classifications * 100))


def main():
    dataset = get_dataset(TRAIN_FILE)
    test_dataset = get_dataset(TEST_FILE)

    network = init_perceptron(dataset)
    network = train_perceptron(network, dataset)
    test_network(network, test_dataset)


if __name__ == '__main__':
    main()