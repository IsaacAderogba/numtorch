from numtorch.operations.Operation import Operation


class TransposeOperation(Operation):
    opcode = "transpose"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self):
        data = self.ctx.data.transpose()

        if self.ctx.meta["autograd"]:
            return self.ctx.tensor(data, {
                "autograd": True,
                "parents": [self.ctx],
                "opcode": self.opcode
            })

        return self.ctx.tensor(data)

    def backward(self, grad):
        parent = self.ctx.meta["parents"][0]
        parent.backward(self.ctx.grad.transpose())
