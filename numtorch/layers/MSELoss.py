from numtorch.layers.Layer import Layer


class MSELoss(Layer):
    def get_params(self):
        return []
        
    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)
