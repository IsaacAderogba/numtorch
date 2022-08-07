# NumTorch, a deep learning framework for building simple neural networks

[NumTorch](https://github.com/IsaacAderogba/numtorch) is a Python library that offers tensor computation through the [NumPy](https://numpy.org/) library and an automatic differentiation system for building dynamic computation graphs.

Created as a learning project from Trask, in Grokking Deep Learning (2019), its API is similar to that of [PyTorch](https://pytorch.org/).

## Docs

#### NumTorch

At a high level, the NumTorch library consists of the following components:

| Component  | Description                                                              |
| ---------- | ------------------------------------------------------------------------ |
| Layers     | Higher level API for composing neural network layers.                    |
| Optimizers | Loss functions for optimizing the weights of a neural network.           |
| Operations | Lower level functions that make up the automatic differentiation system. |
| Tensor     | Core data type that abstracts vectors and matrices.                      |

Tensors act as the core building block, while operations, optimizers, and layers are built on top of them.

#### Layers

Layers are a high level API for defining neural network architectures. A simple neural network, for example, may consist of two linear layers and a layer for calculating the network’s loss.

This can be defined in NumTorch as follows:

```python
import numtorch as nt

model = nt.layers.SequentialLayer([
    nt.layers.LinearLayer(2, 3),
    nt.layers.LinearLayer(3, 1)
])

criterion = nt.layers.MSELoss()
```

You can create your own layers by extending the `Layer` class. Here's a more involved example showcasing a recurrent neural network:

```python
class RNN(nt.layers.Layer):
    def __init__(self, num_inputs, num_hidden, num_outputs, activation=None):
        if activation is None:
            raise Exception("Activation layer expected")

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.activation = activation

        self.weight_input_hidden = nt.layers.LinearLayer(
            num_inputs, num_hidden)
        self.weight_hidden_hidden = nt.layers.LinearLayer(
            num_hidden, num_hidden)
        self.weight_hidden_output = nt.layers.LinearLayer(
            num_hidden, num_outputs)

        self.params = self.weight_input_hidden.get_params()
        self.params += self.weight_hidden_hidden.get_params()
        self.params += self.weight_hidden_output.get_params()

    def init_hidden(self, batch_size=1):
        return nt.Tensor(np.zeros((batch_size, self.num_hidden)), {"autograd": True})

    def get_params(self):
        return self.params

    def forward(self, input, hidden):
        hidden_hidden = self.weight_hidden_hidden.forward(hidden)
        input_hidden = self.weight_input_hidden.forward(input)
        next_hidden = self.activation.forward(input_hidden + hidden_hidden)
        hidden_output = self.weight_hidden_output.forward(next_hidden)

        return hidden_output, next_hidden
```

#### Optimizers

Optimizers are used to update the weights of the network after a training pass. Stochastic Gradient Descent is a popular optimizer, and it can be used to optimizer a network as follows:

```python
optimizer = nt.optimizers.SGDOptimizer(params=model.get_params(), lr=1)

for i in range(10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward()
    optimizer.step()

    print(f"Epoch {i}: {loss}")
```

#### Operations

Operations are the building blocks for NumTorch’s autogradient system. Each operation consists of a forward and backward pass. The forward pass implements the conventional operation, like addition, while the backward pass implements the derivative of that operation.

During the course of training a network, the derivative operations are accumulated to create a dynamic computation graph.

Common operations are exposed on the `Tensor` object:

```python
x = nt.Tensor(np.array([1, 2, 3]), {"autograd": True})

y = x.sum()

y.backward()
```

#### Tensor

A tensor is a data structure that abstracts vectors and matrices from Linear Algebra. Inspired by PyTorch’s tensors, NumTorch’s tensors accumulate gradients to create an automatic differentiation system.

To opt out of the autogradient behaviour, Tensors should simply be initialized with an `autograd` value of `False`.

```python
target = nt.Tensor(np.array([[0], [1], [0], [1]]), {"autograd": False})
```

