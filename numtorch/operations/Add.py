class Add(object):
    def __init__(self, ctx):
        self.ctx = ctx

    def forward(self, other):
        return self.ctx.tensor(self.ctx.data + other.data)
