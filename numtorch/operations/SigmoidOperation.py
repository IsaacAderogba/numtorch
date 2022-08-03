import numpy as np
from numtorch.operations.Operation import Operation


class SigmoidOperation(Operation):
    opcode = "sigmoid"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self):
        data = 1 / (1 + np.exp(-self.ctx.data))

        if self.ctx.meta["autograd"]:
            return self.ctx.tensor(data, {
                "autograd": True,
                "parents": [self.ctx],
                "opcode": self.opcode
            })

        return self.ctx.tensor(data)

    def backward(self, grad):
        parent = self.ctx.meta["parents"][0]
        ones = self.ctx.tensor(np.ones_like(self.ctx.grad.data))

        parent.backward(self.ctx.grad * (self.ctx * (ones - self.ctx)))
