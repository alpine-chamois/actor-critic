import torch
from torch import nn, Tensor


# Two-headed neural network for actor (policy) and critic (value) functions
class ActorCriticAgent(nn.Module):
    OBSERVATION_SPACE_SIZE: int = 4  # Cart position, cart velocity, pole angle and pole angular velocity
    ACTION_SPACE_SIZE: int = 2  # Push cart left and push cart right

    # Configure layers
    def __init__(self) -> None:
        super(ActorCriticAgent, self).__init__()

        self.input_layer = nn.Linear(self.OBSERVATION_SPACE_SIZE, 64)
        self.hidden_layer_1 = nn.Linear(64, 128)

        # Actor (4/25/50/2)
        self.actor_output_layer = nn.Linear(128, self.ACTION_SPACE_SIZE)

        # Critic (4/25/50/25/1)
        self.hidden_layer_2 = nn.Linear(128, 64)
        self.critic_output_layer = nn.Linear(64, 1)

    # Forward propagation
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Normalise the input so the state values are all within the same range
        normalised_input = nn.functional.normalize(x, dim=0)

        # Forward propagation
        input_layer_output = nn.functional.relu(self.input_layer(normalised_input))
        hidden_layer_1_output = nn.functional.relu(self.hidden_layer_1(input_layer_output))

        # The actor head outputs log probabilities over the 2 actions
        actor_output_layer_output = nn.functional.log_softmax(self.actor_output_layer(hidden_layer_1_output), dim=0)

        # Detach the first hidden critic layer to stop further backpropagation through both the actor and critic heads
        hidden_layer_2_output = nn.functional.relu(self.hidden_layer_2(hidden_layer_1_output.detach()))

        # The critic returns the value between -1 and 1
        critic_output_layer_output = torch.tanh(self.critic_output_layer(hidden_layer_2_output))

        # Return output
        return actor_output_layer_output, critic_output_layer_output
