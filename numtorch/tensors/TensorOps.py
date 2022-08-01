from numtorch.operations.AddOperation import AddOperation
from numtorch.operations.CrossEntropyOperation import CrossEntropyOperation
from numtorch.operations.DotProductOperation import DotProductOperation
from numtorch.operations.ExpandOperation import ExpandOperation
from numtorch.operations.IndexOperation import IndexOperation
from numtorch.operations.MultiplyOperation import MultiplyOperation
from numtorch.operations.NegateOperation import NegateOperation
from numtorch.operations.ReLUOperation import ReLUOperation
from numtorch.operations.SigmoidOperation import SigmoidOperation
from numtorch.operations.SubtractOperation import SubtractOperation
from numtorch.operations.SumOperation import SumOperation
from numtorch.operations.TanhOperation import TanhOperation
from numtorch.operations.TransposeOperation import TransposeOperation


class TensorOps(object):
    def __init__(self, tensor):
        self.add = AddOperation(tensor)
        self.neg = NegateOperation(tensor)
        self.sub = SubtractOperation(tensor)
        self.mul = MultiplyOperation(tensor)
        self.sum = SumOperation(tensor)
        self.expand = ExpandOperation(tensor)
        self.transpose = TransposeOperation(tensor)
        self.dot = DotProductOperation(tensor)
        self.sigmoid = SigmoidOperation(tensor)
        self.tanh = TanhOperation(tensor)
        self.index = IndexOperation(tensor)
        self.cross_entropy = CrossEntropyOperation(tensor)
        self.relu = ReLUOperation(tensor)
