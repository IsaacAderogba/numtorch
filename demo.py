import numpy as np
from numtorch.tensors import Tensor

x = Tensor([1, 2, 3, 4, 5])
y = Tensor([2, 2, 2, 2, 2])

z = x + y
z.backward(Tensor(np.array([1, 1, 1, 1, 1])))

print(x.grad)
print(y.grad)
print(z.meta["parents"])
print(z.meta["opcode"])
