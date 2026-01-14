import numpy as np
import itertools

class Parameter:
    __slots__ = ("data", "grad", "_id")

    _id_counter = itertools.count()

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)
        self._id = next(Parameter._id_counter)

    def zero_grad(self):
        self.grad.fill(0.0)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, Parameter) and self._id == other._id
