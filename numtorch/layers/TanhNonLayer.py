from numtorch.layers.Layer import Layer


class TanhNonLayer(Layer):
    def get_params(self):
        return []
        
    def forward(self, input):
        return input.tanh()
