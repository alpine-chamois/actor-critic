import torch
from torch import nn, Tensor


class ActorCriticModel(nn.Module):
    """
    Two-headed neural network for actor (policy) and critic (value) functions
    """

    def __init__(self, observations: int, actions: int) -> None:
        """
        Constructor
        :param observations: the observation space size
        :param actions: the action space size
        """

        super(ActorCriticModel, self).__init__()

        self.input_layer = nn.Linear(observations, 64)
        self.hidden_layer_1 = nn.Linear(64, 128)

        # Actor (4/64/128/2)
        self.actor_output_layer = nn.Linear(128, actions)

        # Critic (4/64/128/64/1)
        self.hidden_layer_2 = nn.Linear(128, 64)
        self.critic_output_layer = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward propagation
        :param x: input tensor
        :return: output tuple (actor tensor, critic tensor)
        """
        # Normalise the input so the state values are all within the same range
        normalised_input = nn.functional.normalize(x, dim=0)

        # Forward propagation
        input_layer_output = nn.functional.relu(self.input_layer(normalised_input))
        hidden_layer_1_output = nn.functional.relu(self.hidden_layer_1(input_layer_output))

        # The actor head outputs logits over the 2 actions
        policy = self.actor_output_layer(hidden_layer_1_output)

        # Detach the first hidden critic layer to stop further backpropagation through both the actor and critic heads
        hidden_layer_2_output = nn.functional.relu(self.hidden_layer_2(hidden_layer_1_output.detach()))

        # The critic returns the value between -1 and 1
        value = torch.tanh(self.critic_output_layer(hidden_layer_2_output))

        # Return output
        return policy, value
