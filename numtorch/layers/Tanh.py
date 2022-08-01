from numtorch.layers.Layer import Layer


class Tanh(Layer):
    def get_params(self):
        return []
        
    def forward(self, input):
        return input.tanh()
