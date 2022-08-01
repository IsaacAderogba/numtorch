from numtorch.layers.Layer import Layer


class CrossEntropyLossLayer(Layer):
    def get_params(self):
        return []

    def forward(self, pred, target):
        return pred.cross_entropy(target)
