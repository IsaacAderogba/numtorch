from numtorch.layers.Layer import Layer


class ReLU(Layer):
    def get_params(self):
        return []
        
    def forward(self, input):
        return input.relu()
