"""
Agent
"""
from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Class to encapsulate an RL agent
    """

    @abstractmethod
    def train(self) -> None:
        """
        Training loop
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Evaluation loop
        """
        pass
