import numtorch as nt
import numpy as np


np.random.seed(0)

data = nt.Tensor(np.array(
    [[0, 0], [0, 1], [1, 0], [1, 1]]),
    {"autograd": True})
target = nt.Tensor(np.array([[0], [1], [0], [1]]), {"autograd": True})

model = nt.layers.SequentialLayer([
    nt.layers.LinearLayer(2, 3),
    nt.layers.Tanh(),
    nt.layers.LinearLayer(3, 1),
    nt.layers.Sigmoid()
])

criterion = nt.layers.MSELoss()
optimizer = nt.optimizers.SGDOptimizer(params=model.get_params(), lr=1)

for i in range(10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward()
    optimizer.step()

    print(f"Epoch {i}: {loss}")
