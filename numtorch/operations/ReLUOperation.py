from numtorch.operations.Operation import Operation


class ReLUOperation(Operation):
    opcode = "relu"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self):
        data = (self.ctx.data > 0) * self.ctx.data

        if self.ctx.meta["autograd"]:
            return self.ctx.tensor(data, {
                "autograd": True,
                "parents": [self.ctx],
                "opcode": self.opcode
            })

        return self.ctx.tensor(data)

    def backward(self, grad):
        parent = self.ctx.meta["parents"][0]
        data = (self.ctx.grad.data > 0) * 1

        parent.backward(self.ctx.tensor(data), self.ctx)
