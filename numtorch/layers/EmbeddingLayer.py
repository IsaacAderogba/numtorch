import numpy as np
from numtorch.tensors.Tensor import Tensor
from numtorch.layers.Layer import Layer


class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, dim):
        self.vocab_size = vocab_size
        self.dim = dim

        self.weight = Tensor(
            (np.random.rand(vocab_size, dim) - 0.5) / dim,
            {"autograd": True}
        )

    def get_params(self):
        return [self.weight]

    def forward(self, input):
        return self.weight.index(input)
