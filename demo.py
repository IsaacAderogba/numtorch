import numpy as np

from numtorch.layers import LinearLayer, EmbeddingLayer, SequentialLayer, TanhNonLinearity, SigmoidNonLinearity, CrossEntropyLoss
from numtorch.tensors import Tensor
from numtorch.optimizers import SGDOptimizer

np.random.seed(0)

data = Tensor(np.array([1, 2, 1, 2]), {"autograd": True})
target = Tensor(np.array([0, 1, 0, 1]), {"autograd": True})

model = SequentialLayer()
model.add(EmbeddingLayer(3, 3))
model.add(TanhNonLinearity())
model.add(LinearLayer(3, 4))

criterion = CrossEntropyLoss()

optimizer = SGDOptimizer(params=model.get_params(), alpha=0.1)

for i in range(10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward()
    optimizer.step()

    print(loss)
