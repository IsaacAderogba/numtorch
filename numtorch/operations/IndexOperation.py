import numpy as np
from numtorch.operations.Operation import Operation


class IndexOperation(Operation):
    opcode = "index"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self, indices):
        data = self.ctx.data[indices.data]

        if self.ctx.meta["autograd"]:
            return self.ctx.tensor(data, {
                "autograd": True,
                "parents": [self.ctx],
                "opcode": self.opcode,
                "indices": indices
            })

        return self.ctx.tensor(data)

    def backward(self, backwardGrad):
        parent = self.ctx.meta["parents"][0]

        data = np.zeros_like(parent.data)
        indices = self.ctx.meta["indices"].data.flatten()
        grad = backwardGrad.data.reshape(len(indices), -1)

        for i in range(len(indices)):
            data[indices[i]] += grad[i]

        parent.backward(self.ctx.tensor(data), self.ctx)
