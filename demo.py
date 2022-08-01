import numpy as np
from numtorch.tensors import Tensor

a = Tensor([
    [1, 2, 3],
    [4, 5, 6],
], {"autograd": True})

b = a.expand(0, 4)
b.backward(Tensor([1, 1, 1, 1, 1]))
print(a.grad)
