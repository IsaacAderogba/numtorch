import numpy as np

from numtorch.operations.Add import Add


class Tensor (object):
    def __init__(self, data) -> None:
        self.data = np.array(data)
        self.ops = {
            "add": Add(self)
        }

    @staticmethod
    def tensor(data):
        return Tensor(data)

    def __add__(self, other):
        return self.ops["add"].forward(other)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
