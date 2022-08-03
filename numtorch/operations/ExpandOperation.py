from numtorch.operations.Operation import Operation


class ExpandOperation(Operation):
    opcode = "expand"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self, dim, copies):
        shape = list(self.ctx.data.shape) + [copies]
        transpose = list(range(0, len(self.ctx.data.shape)))
        transpose.insert(dim, len(self.ctx.data.shape))

        data = self.ctx.data.repeat(copies).reshape(shape).transpose(transpose)

        if self.ctx.meta["autograd"]:
            return self.ctx.tensor(data, {
                "autograd": True,
                "parents": [self.ctx],
                "opcode": self.opcode,
                "dim": dim,
                "copies": copies
            })

        return self.ctx.tensor(data)

    def backward(self, grad):
        parent = self.ctx.meta["parents"][0]
        dim = self.ctx.meta["dim"]

        parent.backward(self.ctx.grad.sum(dim))
