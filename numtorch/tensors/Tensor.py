import numpy as np


class Tensor (object):
    def __init__(self, data) -> None:
        self.data = np.array(data)
