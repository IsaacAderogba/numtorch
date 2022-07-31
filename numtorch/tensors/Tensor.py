import numpy as np

from numtorch.operations.Add import Add


class Tensor (object):
    def __init__(self, data, meta={}) -> None:
        self.data = np.array(data)
        self.grad = None
        self.meta = {"opcode": None, "parents": None, **meta}

        self.ops = {
            "add": Add(self)
        }

    @staticmethod
    def tensor(data, meta={}):
        return Tensor(data, meta)

    def backward(self, grad):
        self.grad = grad

        for key, op in self.ops.items():
            if self.meta["opcode"] is not None and key in self.meta["opcode"]:
                op.backward(grad)

    def __add__(self, other):
        return self.ops["add"].forward(other)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
