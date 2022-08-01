from numtorch.layers.Layer import Layer


class MSELossLayer(Layer):
    def get_params(self):
        return []
        
    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)
