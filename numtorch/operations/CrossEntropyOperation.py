import numpy as np
from numtorch.operations.Operation import Operation


class CrossEntropyOperation(Operation):
    opcode = "cross_entropy"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self, indices):
        temp = np.exp(self.ctx.data)
        softmax_output = temp / np.sum(
            temp, axis=len(self.ctx.data.shape) - 1,
            keepdims=True
        )

        t = indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if self.ctx.meta["autograd"]:
            return self.ctx.tensor(loss, {
                "autograd": True,
                "parents": [self.ctx],
                "opcode": self.opcode,
                "softmax_output": softmax_output,
                "target_dist": target_dist
            })

        return self.ctx.tensor(loss)

    def backward(self, grad):
        parent = self.ctx.meta["parents"][0]
        data = self.ctx.meta["softmax_output"] - self.ctx.meta["target_dist"]
        parent.backward(self.ctx.tensor(data), self.ctx)
