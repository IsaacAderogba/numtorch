from numtorch.operations.Operation import Operation


class MultiplyOperation(Operation):
    opcode = "mul"
    ctx = None

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self, other):
        data = self.ctx.data * other.data

        if self.ctx.meta["autograd"] and other.meta["autograd"]:
            return self.ctx.tensor(data, {
                "autograd": True,
                "parents": [self.ctx, other],
                "opcode": self.opcode
            })

        return self.ctx.tensor(data)

    def backward(self, grad):
        left, right = self.ctx.meta["parents"]
        left.backward(self.ctx.grad * right, self.ctx)
        right.backward(self.ctx.grad * left, self.ctx)
