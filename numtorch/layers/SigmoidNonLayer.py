from numtorch.layers.Layer import Layer


class SigmoidNonLayer(Layer):
    def get_params(self):
        return []
        
    def forward(self, input):
        return input.sigmoid()
