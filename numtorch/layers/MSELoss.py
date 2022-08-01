from numtorch.layers.Layer import Layer


class MSELoss(Layer):
    def get_params(self):
        return []
        
    def forward(self, pred, target):
        error = pred - target
        return (error * error).sum(0)
