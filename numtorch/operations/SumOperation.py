from numtorch.operations.Operation import Operation


class SumOperation(Operation):
    opcode = "sum"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self, dim):
        data = self.ctx.data.sum(dim)

        if self.ctx.meta["autograd"]:
            return self.ctx.tensor(data, {
                "autograd": True,
                "parents": [self.ctx],
                "opcode": self.opcode,
                "dim": dim
            })

        return self.ctx.tensor(data)

    def backward(self, grad):
        parent = self.ctx.meta["parents"][0]

        dim = self.ctx.meta["dim"]
        copies = parent.data.shape[dim]

        parent.backward(self.ctx.grad.expand(dim, copies), self.ctx)
