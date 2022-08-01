import numpy as np
from numtorch.tensors.Tensor import Tensor
from numtorch.layers.Layer import Layer


class LinearLayer(Layer):
    def __init__(self, num_inputs, num_outputs):
        weight = np.random.randn(num_inputs, num_outputs)

        self.weight = Tensor(
            weight * np.sqrt(2.0 / num_inputs),
            {"autograd": True}
        )

        self.bias = Tensor(np.zeros(num_outputs), {"autograd": True})

    def get_params(self):
        return [self.weight, self.bias]

    def forward(self, input):
        return input.dot(self.weight) + self.bias.expand(0, len(input.data))
