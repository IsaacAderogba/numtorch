import numpy as np
from numtorch.tensors import Tensor

a = Tensor([[1, 2, 3, 4, 5]], {"autograd": True})

b = a.transpose()
print(b)
b.backward(Tensor([[1], [2], [3], [4], [5]]))
print(a.grad)
