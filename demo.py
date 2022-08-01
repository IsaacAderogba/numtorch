import numpy as np
from numtorch.tensors import Tensor

a = Tensor([1, 2, 3, 4, 5], {"autograd": True})
b = Tensor([2, 2, 2, 2, 2], {"autograd": True})
c = Tensor([5, 4, 3, 2, 1], {"autograd": True})

d = a.dot(b)
e = b.dot(c)
f = d.dot(e)

print(f)
f.backward(Tensor([1, 1, 1, 1, 1]));

print(b.grad)
