import math
import sys
import gym
import matplotlib.pyplot as pyplot
import numpy
# noinspection PyUnresolvedReferences
import pygame  # Required for OpenAI Gym rendering
import torch
from torch import nn, Tensor
from actor_critic_agent import ActorCriticAgent

# Training and evaluation parameters
EPISODES: int = 200
LEARNING_STEPS: int = 10
LEARNING_RATE: float = 0.001
REWARD_DISCOUNT_FACTOR: float = 0.95
CRITIC_LOSS_SCALING: float = 0.1
MODEL_FILE: str = 'a2c.mdl'
TRAIN_MODEL: bool = True
EVALUATION_STEPS: int = 500
AVERAGING_WINDOW: int = round(EPISODES / 10)
TRAINING_OBSERVATION_NOISE_RATIO: float = 0.0  # Perfect training
EVALUATION_OBSERVATION_NOISE_RATIO: float = 0.1  # Realistic evaluation


# Training loop
def train() -> None:
    # Create environment and agent
    environment: gym.Env = gym.make('CartPole-v1')
    agent: nn.Module = ActorCriticAgent()

    # Create a gradient descent optimiser with a learning rate of 0.001
    optimizer: torch.optim.Optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    # Store episode metrics
    episode_durations: list[int] = []
    losses: list[float] = []

    for episode in range(EPISODES):

        # Print training progress
        print_training_progress(episode)

        # Reset the environment and observe initial state
        observation, info = environment.reset()

        # Initialise episode data
        terminated: bool = False
        truncated: bool = False
        values: list[Tensor] = []
        log_action_probabilities: list[Tensor] = []
        rewards: list[int] = []
        episode_duration: int = 0
        step: int = 0
        latest_value: Tensor = torch.Tensor([0])

        while not terminated and not truncated:

            # Run the agent
            policy, value = agent(torch.from_numpy(observation).float())

            # Determine the action
            logits = policy.view(-1)
            action_distribution = torch.distributions.Categorical(logits=logits)
            action = action_distribution.sample()

            # Store the value and action probability
            values.append(value)
            log_action_probability = policy.view(-1)[action]
            log_action_probabilities.append(log_action_probability)

            # Perform the action on the environment, receive a reward and observe the new state
            observation, reward, terminated, truncated, info = environment.step(action.detach().numpy())

            # Add observation noise
            observation = add_observation_noise(observation, TRAINING_OBSERVATION_NOISE_RATIO)

            # Increment the step and episode duration
            step += 1
            episode_duration += 1

            # Penalise the agent for losing the game
            if terminated:
                reward = -10
            # Otherwise record the last value
            else:
                latest_value = value

            # Store the reward
            rewards.append(reward)

            # Optimise after N steps or at the end of the episode
            if step == LEARNING_STEPS or terminated or truncated:
                loss = optimise(agent, optimizer, log_action_probabilities, rewards, values, latest_value)
                losses.append(loss)
                # Reset the optimisation data
                values, log_action_probabilities, rewards = [], [], []
                step = 0

        # Store the episode length
        episode_durations.append(episode_duration)

    # Plot training metrics
    plot_training_metrics(episode_durations, losses)

    # Save the agent
    torch.save(agent.state_dict(), MODEL_FILE)


# Optimise the agent
def optimise(agent: nn.Module, optimizer: torch.optim.Optimizer, log_action_probabilities: list[Tensor],
             rewards: list[int], values: list[Tensor], latest_value: Tensor) -> float:
    # Reverse the episode data so most recent data comes first in 1D tensors
    rewards_: Tensor = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    log_action_probabilities_: Tensor = torch.stack(log_action_probabilities).flip(dims=(0,)).view(-1)
    values_: Tensor = torch.stack(values).flip(dims=(0,)).view(-1)

    # Calculate weighted rewards
    returns: list[Tensor] = []
    # Bootstrap the latest value
    return_: Tensor = latest_value.detach()  # Detach to stop double backpropagation
    for reward in range(rewards_.shape[0]):
        # Discount older rewards
        return_ = rewards_[reward] + REWARD_DISCOUNT_FACTOR * return_
        returns.append(return_)
    # Make tensor 1D
    returns_: Tensor = torch.stack(returns).view(-1)
    # Normalise rewards between -1 and 1
    returns_ = torch.nn.functional.normalize(returns_, dim=0)

    # Calculate advantage and loss
    advantage: Tensor = returns_ - values_.detach()  # Detach to stop double backpropagation
    actor_loss: Tensor = -1 * log_action_probabilities_ * advantage
    critic_loss: Tensor = torch.pow(values_ - returns_, 2)

    # Scale down the critic loss because we want the actor to learn faster than the critic
    loss: Tensor = actor_loss.sum() + CRITIC_LOSS_SCALING * critic_loss.sum()

    # Calculate gradients
    agent.zero_grad()
    optimizer.zero_grad()
    loss.backward()

    # Optimise
    optimizer.step()

    return loss.detach().numpy()


# Evaluation loop
def evaluate() -> None:
    # Create environment and agent
    environment: gym.Env = gym.make('CartPole-v1', render_mode='human')
    agent: nn.Module = ActorCriticAgent()

    # Load the trained agent parameters
    agent.load_state_dict(torch.load(MODEL_FILE))

    # Put the agent in evaluation mode
    agent.eval()

    # Reset the environment and observer the initial state
    observation, info = environment.reset()

    # Play the game
    terminated: bool = False
    step: int = 0
    while not terminated and step < EVALUATION_STEPS:
        step += 1

        # Run the agent
        policy, value = agent(torch.from_numpy(observation).float())

        # Make the tensor 1D
        logits = policy.view(-1)
        # Determine the best action
        action = logits.detach().numpy().argmax()

        # Perform the action on the environment and observe the new state
        observation, reward, terminated, truncated, info = environment.step(action)

        # Add observation noise
        observation = add_observation_noise(observation, EVALUATION_OBSERVATION_NOISE_RATIO)

        # Render the step
        environment.render()


# Add observation noise
def add_observation_noise(observation: numpy.ndarray, noise_ratio: float) -> numpy.array:
    noisy_observation: list[float] = []
    for state in observation:
        noise = numpy.random.normal(0.0, math.fabs(state * noise_ratio))
        noisy_observation.append(state + noise)
    return numpy.array(noisy_observation)


# Print progress to terminal
def print_training_progress(episode: int) -> None:
    percentage: int = math.ceil(((episode + 1) / EPISODES) * 100)
    sys.stdout.write('\rTraining (' + str(percentage) + '%)')
    sys.stdout.flush()


# Plot training metrics
def plot_training_metrics(episode_durations: list[int], losses: list[float]) -> None:
    pyplot.suptitle('Training Metrics')
    pyplot.subplot(1, 2, 1)
    pyplot.plot(average_metrics(episode_durations), color='green')
    pyplot.xlabel('Episode')
    pyplot.ylabel('Duration')
    pyplot.subplot(1, 2, 2)
    pyplot.plot(average_metrics(losses), color='red')
    pyplot.xlabel('Optimisation Step')
    pyplot.ylabel('Loss')
    pyplot.show()


# Average metrics for plotting
def average_metrics(metrics: list[[int | float]]) -> list[numpy.ndarray]:
    average: list[any] = []
    for index in range(AVERAGING_WINDOW - 1):
        average.append(numpy.nan)
    for index in range(len(metrics) - AVERAGING_WINDOW + 1):
        average.append(numpy.mean(metrics[index:index + AVERAGING_WINDOW]))
    return average


# Main function
if __name__ == '__main__':

    if TRAIN_MODEL is True:
        train()
    evaluate()
