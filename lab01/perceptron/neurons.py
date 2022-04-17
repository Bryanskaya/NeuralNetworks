from typing import Callable, Optional, Union, List


class Neuron:
    def __init__(self, f_initialize: Optional[Callable[[], int]]):
        self.input_weights = []
        self.bias = 0
        self.initialize = f_initialize

    def init_weights(self, count: int) -> None:
        for _ in range(count):
            self.input_weights.append(self.initialize())
        self.bias = self.initialize()

    def reinit_weights(self) -> None:
        self.input_weights = [self.initialize() for _ in self.input_weights]
        self.bias = self.initialize()

    def solve(self, inputs: List[int]) -> List[int]:
        raise NotImplementedError

    def correct(self, expected_result: int) -> None:
        pass


class ActivationNeuron(Neuron):
    def __init__(self, f_initialize, f_activate):
        # 1: None
        # 2: Callable[[int], int]
        super().__init__(f_initialize)
        self.last_inputs = None
        self.last_result = None
        self.activate = f_activate

    def accumulate(self, inputs: List[int]) -> float:
        accumulation = - self.bias
        for value, weight in zip(inputs, self.input_weights):
            accumulation += value * weight
        return accumulation

    def solve(self, inputs: List[int]) -> List[int]:
        self.last_inputs = inputs
        self.last_result = self.activate(self.accumulate(inputs))
        return self.last_result


class SNeuron(Neuron):
    def __init__(self, f_initialize, f_transform: Callable[[List[int]], int]):
        super().__init__(f_initialize)
        self.transform = f_transform

    def solve(self, inputs: List[int]) -> int:
        return self.transform(inputs)


class ANeuron(ActivationNeuron):
    def calculate_bias(self) -> None:
        self.bias = 0
        for weight in self.input_weights:
            if weight > 0:  self.bias += 1
            elif weight < 0:    self.bias -= 1


class RNeuron(ActivationNeuron):
    def __init__(self, f_initialize, f_activate, learning_speed, bias):
        super().__init__(f_initialize, f_activate)
        self.learning_speed = learning_speed
        self.bias = bias

    def correct(self, expected_result: int) -> None:
        if expected_result != self.last_result:
            self.input_weights = [
                input_weight - self.last_result * self.learning_speed * last_input
                for input_weight, last_input in zip(self.input_weights, self.last_inputs)
            ]
            self.bias += self.last_result * self.learning_speed
