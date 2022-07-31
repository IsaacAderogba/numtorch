class Add(object):
    opcode = "add"

    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self, other):
        return self.ctx.tensor(self.ctx.data + other.data, {
            "parents": [self.ctx, other],
            "opcode": self.opcode
        })

    def backward(self, grad):
        left, right = self.ctx.meta["parents"]
        left.backward(self.ctx.tensor(self.ctx.grad.data))
        right.backward(self.ctx.tensor(self.ctx.grad.data))
