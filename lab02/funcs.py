import numpy as np

from abc import ABC


class Func(ABC):
    @staticmethod
    def f(z):
        pass

    @staticmethod
    def df(z):
        pass


class TanFunc(Func):
    @staticmethod
    def f(z):
        return np.tanh(z)

    @staticmethod
    def df(z):
        return 1 - z ** 2


class ReLUFunc(Func):
    @staticmethod
    def f(z):
        return np.maximum(0, z)

    @staticmethod
    def df(z):
        return (z > 0) * 1


class SigmoidFunc(Func):
    @staticmethod
    def f(z):
        return 1. / (1. + np.exp(-z))

    @staticmethod
    def df(z):
        return SigmoidFunc.f(z) * (1 - SigmoidFunc.f(z))
