import uuid
import numpy as np

from numtorch.operations.AddOperation import AddOperation
from numtorch.operations.ExpandOperation import ExpandOperation
from numtorch.operations.MultiplyOperation import MultiplyOperation
from numtorch.operations.NegateOperation import NegateOperation
from numtorch.operations.SubtractOperation import SubtractOperation
from numtorch.operations.SumOperation import SumOperation


class Tensor (object):
    def __init__(self, data, meta={}):
        self.id = str(uuid.uuid4())
        self.data = np.array(data)
        self.grad = None

        self.ops = {
            "add": AddOperation(self),
            "neg": NegateOperation(self),
            "sub": SubtractOperation(self),
            "mul": MultiplyOperation(self),
            "sum": SumOperation(self),
            "expand": ExpandOperation(self)
        }

        self.meta = {
            "opcode": "",
            "parents": [],
            "autograd": False,
            "children": {},
            **meta
        }

        for parent in self.meta["parents"]:
            if(self.id not in parent.meta["children"]):
                parent.meta["children"][self.id] = 1
            else:
                parent.meta["children"][self.id] += 1

    @staticmethod
    def tensor(data, meta={}):
        return Tensor(data, meta)

    def have_grads_accumulated(self):
        for count in self.meta["children"].values():
            if count != 0:
                return False

        return True

    def backward(self, grad=None, ctx=None):
        if self.meta["autograd"] == False:
            return None

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        assert grad.meta["autograd"] == False

        if ctx is not None:
            if self.meta["children"][ctx.id] == 0:
                raise Exception("cannot backprop more than once")

            self.meta["children"][ctx.id] -= 1

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self.meta["parents"] is not None and (self.have_grads_accumulated() or ctx is None):
            for key, op in self.ops.items():
                if key == self.meta["opcode"]:
                    op.backward(grad)

    def __add__(self, other):
        return self.ops["add"].forward(other)

    def __sub__(self, other):
        return self.ops["sub"].forward(other)

    def __neg__(self):
        return self.ops["neg"].forward()

    def __mul__(self, other):
        return self.ops["mul"].forward(other)

    def sum(self, dim):
        return self.ops["sum"].forward(dim)

    def expand(self, dim, copies):
        return self.ops["expand"].forward(dim, copies)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
