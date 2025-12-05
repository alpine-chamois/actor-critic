import math
import sys
from typing import SupportsFloat

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy

# noinspection PyUnresolvedReferences
import pygame  # Required for rendering # noqa
import torch
from gymnasium.core import ObsType
from torch import nn, Tensor

from actorcritic.actor_critic_model import ActorCriticModel
from actorcritic.agent import Agent


class ActorCriticAgent(Agent):
    """
    Class to encapsulate an Actor-Critic agent
    """

    def __init__(self, game: str, max_cumulative_reward: int):
        """
        Constructor
        :param game: the environment name
        :param max_cumulative_reward: the total rewards to terminate training on
        """

        super(ActorCriticAgent, self).__init__()

        # Training and evaluation parameters
        self.game: str = game
        self.max_cumulative_reward: float = max_cumulative_reward
        self.n_steps: int = 10  # Steps between optimisations
        self.learning_rate: float = 0.0005
        self.gamma: float = 0.95  # The Bellman discount factor
        self.critic_loss_scaling: float = 0.5
        self.entropy_coefficient: float = 0.0001
        self.model_file: str = "a2c.mdl"
        self.averaging_window: int = 10

    def train(self) -> None:
        """
        Training loop
        """
        # Create environment and agent
        environment: gym.Env = gym.make(self.game)
        observations: int = environment.observation_space.shape[0]  # type: ignore
        actions: int = environment.action_space.n  # type: ignore
        model: nn.Module = ActorCriticModel(observations=observations, actions=actions)

        # Put model in train mode
        model.train()

        # Create a gradient descent optimiser
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Store episode metrics
        episode: int = 0
        cumulative_rewards: list[float] = []
        average_cumulative_rewards: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropy_losses: list[float] = []

        while self._is_below_performance_threshold(average_cumulative_rewards):
            # Reset the environment and observe initial state
            observation, info = environment.reset()

            # Initialise training variables
            step: int = 0
            terminated: bool = False
            truncated: bool = False
            cumulative_reward: float = 0
            predicted_next_value: Tensor = torch.tensor([0.0])

            # Initialise rollout buffers
            predicted_values: list[Tensor] = []
            log_action_probabilities: list[Tensor] = []
            rewards: list[SupportsFloat] = []
            entropies: list[Tensor] = []

            while not terminated and not truncated:
                # Run the agent
                policy, value = model(torch.from_numpy(observation).float())

                # Determine the action
                logits = policy.view(-1)
                action_distribution = torch.distributions.Categorical(logits=logits)
                action = action_distribution.sample()

                # Store the value and log probability
                predicted_values.append(value)
                log_action_probability = action_distribution.log_prob(action)
                log_action_probabilities.append(log_action_probability)

                # Store the entropy
                entropy = action_distribution.entropy()
                entropies.append(entropy)

                # Perform the action on the environment, receive a reward and observe the new state
                observation, reward, terminated, truncated, info = environment.step(action.item())

                # Reward shaping
                reward = self._shape_reward(observation, reward, terminated, truncated)

                # Increment the step and cumulative reward
                step += 1
                cumulative_reward += reward  # type: ignore

                # Store the reward
                rewards.append(reward)

                # Optimise after N steps or at the end of the episode
                if step == self.n_steps or terminated or truncated:
                    # Record the predicted value of the next state if not terminated
                    if not terminated:
                        with torch.no_grad():
                            _, next_value = model(torch.from_numpy(observation).float())
                            predicted_next_value = next_value.detach()  # Detach to stop double backpropagation
                    else:
                        predicted_next_value = torch.tensor([0.0])

                    # Optimise
                    actor_loss, critic_loss, entropy_loss = self._optimise(
                        model,
                        optimizer,
                        log_action_probabilities,
                        rewards,
                        predicted_values,
                        predicted_next_value,
                        entropies,
                    )
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    entropy_losses.append(entropy_loss)

                    # Reset the rollout buffers and training variables
                    predicted_values, log_action_probabilities, rewards, entropies = (
                        [],
                        [],
                        [],
                        [],
                    )
                    predicted_next_value = torch.tensor([0.0])
                    step = 0

            # Increment the episode
            episode += 1

            # Store the cumulative reward
            cumulative_rewards.append(cumulative_reward)
            average_cumulative_rewards = self._average_metrics(cumulative_rewards)

            # Print training progress
            self._print_training_progress(episode, average_cumulative_rewards)

        # Plot training metrics
        average_actor_losses: list[float] = self._average_metrics(actor_losses)
        average_critic_losses: list[float] = self._average_metrics(critic_losses)
        average_entropy_losses: list[float] = self._average_metrics(entropy_losses)
        self._plot_training_metrics(
            average_cumulative_rewards,
            average_actor_losses,
            average_critic_losses,
            average_entropy_losses,
        )

        # Save the agent
        torch.save(model.state_dict(), self.model_file)

        # Close the environment
        environment.close()

    def _optimise(
        self,
        agent: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_action_probabilities: list[Tensor],
        rewards: list[SupportsFloat],
        values: list[Tensor],
        predicted_next_value: Tensor,
        entropies: list[Tensor],
    ) -> tuple[float, float, float]:
        """
        Optimise
        :param agent: the agent
        :param optimizer: the optimiser
        :param log_action_probabilities: the log action probabilities
        :param rewards: the rewards
        :param values: the values
        :param predicted_next_value: the predicted value to bootstrap
        :param entropies: the entropies
        :return: the actor loss, critic_loss and entropy_loss
        """
        # Convert lists to tensors
        rewards_: Tensor = torch.tensor(rewards).view(-1)
        log_action_probabilities_: Tensor = torch.stack(log_action_probabilities).view(-1)
        values_: Tensor = torch.stack(values).view(-1)
        entropies_: Tensor = torch.stack(entropies).view(-1)

        # Apply the Bellman equation to calculate the value of the current state
        returns: list[Tensor] = []
        return_: Tensor = predicted_next_value.clone()
        for reward in reversed(rewards_):
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)
        returns_: Tensor = torch.stack(returns).view(-1)
        # Normalise returns between -1 and 1
        returns_ = torch.nn.functional.normalize(returns_, dim=0)

        # Calculate advantage and loss
        advantage: Tensor = returns_ - values_.detach()  # Detach to stop double backpropagation
        actor_loss: Tensor = (-1 * log_action_probabilities_ * advantage).sum()
        critic_loss: Tensor = torch.pow(values_ - returns_, 2).sum()
        entropy_loss: Tensor = -entropies_.sum()

        # Scale down the critic loss because we want the actor to learn faster than the critic
        # Entropy loss is negative as we want to encourage exploration
        loss: Tensor = actor_loss + (self.critic_loss_scaling * critic_loss) + (self.entropy_coefficient * entropy_loss)

        # Calculate gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimise
        optimizer.step()

        return (
            actor_loss.item(),
            critic_loss.item(),
            entropy_loss.item(),
        )

    def evaluate(self, render: bool) -> None:
        """
        Evaluation loop
        :param render: whether to render or not
        """
        # Set render mode
        render_mode: str = "human"
        if not render:
            render_mode = "rgb_array"

        # Create environment and agent
        environment: gym.Env = gym.make(self.game, render_mode=render_mode)
        observations: int = environment.observation_space.shape[0]  # type: ignore
        actions: int = environment.action_space.n  # type: ignore
        model: nn.Module = ActorCriticModel(observations, actions)

        # Load the trained agent parameters
        model.load_state_dict(torch.load(self.model_file))

        # Put the agent in evaluation mode
        model.eval()

        # Reset the environment and observe the initial state
        observation, info = environment.reset()

        # Play the game
        truncated: bool = False
        terminated: bool = False
        while not terminated and not truncated:
            # Run the agent
            policy, value = model(torch.from_numpy(observation).float())

            # Make the tensor 1D
            logits = policy.view(-1)
            # Determine the best action
            action = logits.detach().numpy().argmax()

            # Perform the action on the environment and observe the new state
            observation, reward, terminated, truncated, info = environment.step(action)

            # Render the step
            environment.render()

        # Close the environment
        environment.close()

    def _shape_reward(
        self,
        _observation: ObsType,  # type: ignore
        reward: SupportsFloat,
        _terminated: bool,
        _truncated: bool,
    ) -> SupportsFloat:
        """
        Shape reward
        :param _observation: the observation
        :param reward: the reward
        :param _terminated: if the episode terminated
        :param _truncated: if the episode truncated
        :return: shaped reward
        """
        return reward

    def _is_below_performance_threshold(self, average_cumulative_rewards: list[float]) -> bool:
        """
        Determine if the performance is below the threshold
        :param average_cumulative_rewards: the average cumulative rewards
        """
        return not average_cumulative_rewards or average_cumulative_rewards[-1] < self.max_cumulative_reward

    def _print_training_progress(self, episode: int, average_cumulative_rewards: list[float]) -> None:
        """
        Print progress to terminal
        :param episode: the episode
        :param average_cumulative_rewards: the average cumulative rewards
        """
        sys.stdout.write("\r\x1b[2K")  # Clear line
        sys.stdout.write(
            "Learning to play "
            + self.game
            + ", average cumulative reward: "
            + str(math.ceil(average_cumulative_rewards[-1]))
            + "/"
            + str(self.max_cumulative_reward)
            + " after "
            + str(episode)
            + " training episodes."
        )
        sys.stdout.flush()

    def _plot_training_metrics(
        self,
        average_cumulative_rewards: list[float],
        average_actor_losses: list[float],
        average_critic_losses: list[float],
        average_entropy_losses: list[float],
    ) -> None:
        """
        Plot training metrics
        :param average_cumulative_rewards: the average cumulative rewards
        :param average_actor_losses: the average actor losses
        :param average_critic_losses: the average critic losses
        :param average_entropy_losses: the average entropy losses
        """
        plt.suptitle("Training metrics")
        plt.subplot(1, 2, 1)
        plt.plot(average_cumulative_rewards, color="tab:cyan")
        plt.fill_between(
            range(len(average_cumulative_rewards)),
            average_cumulative_rewards,
            color="tab:cyan",
            alpha=0.5,
        )
        plt.xlim(left=self.averaging_window)
        plt.xlabel("Episode")
        plt.ylabel("Average cumulative reward")
        plt.subplot(1, 2, 2)
        plt.plot(average_actor_losses, color="tab:cyan", alpha=1.0, label="Actor")
        plt.plot(average_critic_losses, color="tab:cyan", alpha=0.6, label="Critic")
        plt.plot(average_entropy_losses, color="tab:cyan", alpha=0.2, label="Entropy")
        plt.xlim(left=self.averaging_window)
        plt.xlabel("Optimisation step")
        plt.ylabel("Average loss")
        plt.legend(loc="upper right")
        plt.show()

    def _average_metrics(self, metrics: list[float]) -> list[float]:
        """
        Average metrics for plotting
        :param metrics: the metrics
        :return: averaged metrics
        """
        # Note that this wil discard rewards before the averaging window is full so the plot Y-axis intercept needs setting
        return list(
            numpy.convolve(
                metrics,
                numpy.ones(self.averaging_window) / self.averaging_window,
                mode="valid",
            )
        )
