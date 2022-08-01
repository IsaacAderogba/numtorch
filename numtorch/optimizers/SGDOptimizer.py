
class SGDOptimizer(object):
    def __init__(self, params, alpha=0.1):
        self.params = params
        self.alpha = alpha

    def zero(self):
        for param in self.params:
            param.grad.data *= 0

    def step(self, zero=True):
        for p in self.params:
            p.data -= p.grad.data * self.alpha

            if (zero):
                p.grad.data *= 0
