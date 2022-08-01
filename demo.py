import numpy as np

from numtorch.layers import LinearLayer, SequentialLayer, MSELossLayer
from numtorch.tensors import Tensor
from numtorch.optimizers import SGDOptimizer

np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), {"autograd": True})
target = Tensor(np.array([[0], [1], [0], [1]]), {"autograd": True})

model = SequentialLayer()
model.add(LinearLayer(2, 3))
model.add(LinearLayer(3, 1))

criterion = MSELossLayer()


optimizer = SGDOptimizer(params=model.get_params(), alpha=0.05)

for i in range(10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward()
    optimizer.step()

    print(loss)
