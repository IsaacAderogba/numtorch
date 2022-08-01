from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def forward(self):
        pass