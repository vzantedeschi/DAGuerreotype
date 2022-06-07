from abc import ABCMeta, abstractmethod

from torch import nn

from modules import Structure, LinearEquations

class Estimator(ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
    	pass

class LinearEstimator(Estimator, nn.Module):

    def __init__(self):

        super(LinearEstimator, self).__init__()

        self.structure = Structure()
        self.equations = LinearEquations()

    @abstractmethod
    def forward(self):
    	pass