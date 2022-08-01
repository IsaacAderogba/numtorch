import numpy as np
from numtorch.tensors import Tensor
from numtorch.optimizers import SDGOptimizer

np.random.seed(0)


data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), {"autograd": True})
target = Tensor(np.array([[0], [1], [0], [1]]), {"autograd": True})

w = list()
w.append(Tensor([[0.71518937, 0.60276338, 0.54488318],
                 [0.4236548, 0.64589411, 0.43758721]], {"autograd": True}))
w.append(Tensor([[0.05671298],
                 [0.27265629],
                 [0.47766512]], {"autograd": True}))

optimizer = SDGOptimizer(params=w, alpha=0.1)

for i in range(10):
    pred = data.dot(w[0]).dot(w[1])
    loss = ((pred - target)*(pred - target)).sum(0)
    loss.backward()
    optimizer.step()

    print(loss)
