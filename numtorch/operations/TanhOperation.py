import numpy as np
from numtorch.operations.Operation import Operation


class TanhOperation(Operation):
    opcode = "tanh"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self):
        data = np.tanh(self.ctx.data)

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

        parent.backward(
            self.ctx.grad * (ones - (self.ctx * self.ctx))
        )
