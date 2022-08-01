from numtorch.layers.Layer import Layer


class SequentialLayer(Layer):
    def __init__(self, layers=list()):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def get_params(self):
        params = list()

        for layer in self.layers:
            params += layer.get_params()

        return params

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)

        return input
