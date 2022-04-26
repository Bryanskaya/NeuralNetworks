from perceptron.neurons import *


class Layer:
    def __init__(self):
        self.neurons = []

    def reinit_weights(self) -> None:
        for neuron in self.neurons:
            neuron.reinit_weights()

    def solve(self, inputs):
        raise NotImplementedError

    def correct(self, expected_results):
        pass


class SLayer(Layer):
    def add_neuron(self, f_initialize, f_transform) -> None:
        neuron = SNeuron(f_initialize, f_transform)
        self.neurons.append(neuron)

    def solve(self, inputs: List[int]) -> List[int]:
        results = []
        for neuron, value in zip(self.neurons, inputs):
            results.append(neuron.solve(value))
        return results


class ALayer(Layer):
    def add_neuron(self, inputs_count, f_initialize, f_activate) -> None:
        neuron = ANeuron(f_initialize, f_activate)
        neuron.init_weights(inputs_count)
        self.neurons.append(neuron)

    def solve(self, inputs: List[int]) -> List[int]:
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results


class RLayer(Layer):
    def add_neuron(self, inputs_count: int, f_initialize, f_activate,
                   learning_speed: float, bias: float) -> None:
        neuron = RNeuron(f_initialize, f_activate, learning_speed, bias)
        neuron.init_weights(inputs_count)
        self.neurons.append(neuron)

    def solve(self, inputs: List[int]) -> List[List[int]]:
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results

    def correct(self, expected_results: List[int]) -> None:
        for neuron, expected_result in zip(self.neurons, expected_results):
            neuron.correct(expected_result)
