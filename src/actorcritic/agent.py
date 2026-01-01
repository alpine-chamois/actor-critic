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
    def evaluate(self, render: bool) -> None:
        """
        Evaluation loop
        :param render: whether to render or not
        """
        pass
