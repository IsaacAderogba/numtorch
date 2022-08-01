from numtorch.layers.Layer import Layer


class CrossEntropyLoss(Layer):
    def get_params(self):
        return []

    def forward(self, pred, target):
        return pred.cross_entropy(target)
