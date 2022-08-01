from numtorch.layers.Layer import Layer


class TanhNonLinearity(Layer):
    def get_params(self):
        return []
        
    def forward(self, input):
        return input.tanh()
