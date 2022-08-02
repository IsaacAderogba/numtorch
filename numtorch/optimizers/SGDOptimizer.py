
class SGDOptimizer(object):
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr

    def zero(self):
        for param in self.params:
            param.grad.data *= 0

    def step(self, zero=True):
        for p in self.params:
            p.data -= p.grad.data * self.lr

            if (zero):
                p.grad.data *= 0
