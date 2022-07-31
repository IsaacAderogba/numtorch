class Add(object):
    opcode = "add"

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self, other):
        data = self.ctx.data + other.data

        if self.ctx.meta["autograd"] and other.meta["autograd"]:
            return self.ctx.tensor(data, {
                "autograd": True,
                "parents": [self.ctx, other],
                "opcode": self.opcode
            })

        return self.ctx.tensor(data)

    def backward(self, grad):
        left, right = self.ctx.meta["parents"]
        left.backward(self.ctx.tensor(self.ctx.grad.data), self.ctx)
        right.backward(self.ctx.tensor(self.ctx.grad.data), self.ctx)
