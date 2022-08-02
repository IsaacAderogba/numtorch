import numpy as np

from numtorch.layers.Layer import Layer
from numtorch.layers.LinearLayer import LinearLayer
from numtorch.tensors.Tensor import Tensor


class RecurrentCell(Layer):
    def __init__(self, num_inputs, num_hidden, num_outputs, activation=None):
        if activation is None:
            raise Exception("Activation layer expected")

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.activation = activation

        self.weight_input_hidden = LinearLayer(num_inputs, num_hidden)
        self.weight_hidden_hidden = LinearLayer(num_hidden, num_hidden)
        self.weight_hidden_output = LinearLayer(num_hidden, num_outputs)

        self.params = self.weight_input_hidden.get_params()
        self.params += self.weight_hidden_hidden.get_params()
        self.params += self.weight_hidden_output.get_params()

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.num_hidden)), {"autograd": True})

    def get_params(self):
        return self.params

    def forward(self, input, hidden):
        input_hidden = self.weight_input_hidden.forward(input)
        hidden_hidden = self.weight_hidden_hidden.forward(hidden)
        next_hidden = self.activation.forward(input_hidden + hidden_hidden)
        hidden_output = self.weight_hidden_hidden_output(next_hidden)

        return hidden_output, next_hidden
