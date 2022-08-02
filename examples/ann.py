import numtorch as nt
import numpy as np


np.random.seed(0)

data = nt.Tensor(np.array([1, 2, 1, 2]), {"autograd": True})
target = nt.Tensor(np.array([0, 1, 0, 1]), {"autograd": True})

model = nt.layers.SequentialLayer()
model.add(nt.layers.EmbeddingLayer(3, 3))
model.add(nt.layers.Tanh())
model.add(nt.layers.LinearLayer(3, 4))

criterion = nt.layers.CrossEntropyLoss()

optimizer = nt.optimizers.SGDOptimizer(params=model.get_params(), lr=0.1)

for i in range(10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward()
    optimizer.step()

    print(f"Epoch {i}: {loss}")
