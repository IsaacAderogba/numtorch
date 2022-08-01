from abc import ABC, abstractmethod

class Operation(ABC):
    @property
    @abstractmethod
    def ctx(self):
        pass

    @property
    @abstractmethod
    def opcode(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, grad):
        pass