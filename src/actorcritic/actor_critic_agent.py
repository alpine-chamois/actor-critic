import torch
import gymnasium as gym

# noinspection PyUnresolvedReferences
import pygame  # Required for rendering # noqa
from stable_baselines3 import A2C
import stable_baselines3.common.on_policy_algorithm as algorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from actorcritic.agent import Agent


class TerminationPenaltyWrapper(gym.Wrapper):
    """
    Penalise the agent for losing the game
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            reward = -10

        return observation, reward, terminated, truncated, info

class ActorCriticAgent(Agent):
    """
    Class to encapsulate an Actor-Critic agent
    """

    def __init__(self, game: str, max_cumulative_reward: float):
        """
        Constructor
        :param game: the environment name
        :param max_cumulative_reward: the total rewards to terminate training on
        """

        super(ActorCriticAgent, self).__init__()

        # Training and evaluation parameters
        self.game: str = game
        self.max_cumulative_reward: float = max_cumulative_reward
        # Specify network parameters to match the PyTorch agent (output layers and activation functions are
        # automatically set)
        # Actor (4/64/128/2)
        # Critic (4/64/128/64/1)
        self.net_arch = dict(pi=[64, 128], vf=[64, 128, 64])
        self.activation_fn = torch.nn.ReLU
        self.n_steps: int = 10  # Steps between optimisations
        self.learning_rate: float = 0.0005
        self.entropy_coefficient: float = 0.0001
        self.gamma: float = 0.95  # The Bellman discount factor
        self.averaging_window: int = 10
        self.eval_freq: int = 1000  # Steps between evaluations
        self.model_file: str = "a2c.mdl"

    def train(self) -> None:
        """
        Training loop
        """
        # Create environment and agent
        training_environment: gym.Env = TerminationPenaltyWrapper(gym.make(self.game))
        evaluation_environment: gym.Env = TerminationPenaltyWrapper(gym.make(self.game))
        policy_kwargs = dict(activation_fn=self.activation_fn, net_arch=self.net_arch)
        agent: A2C = A2C(
            "MlpPolicy",
            training_environment,
            policy_kwargs=policy_kwargs,
            n_steps=self.n_steps,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            verbose=1,
            ent_coef=self.entropy_coefficient,
            tensorboard_log="logs",
        )

        # Train the agent
        callback_on_best: BaseCallback = StopTrainingOnRewardThreshold(
            reward_threshold=self.max_cumulative_reward, verbose=1
        )
        eval_callback: BaseCallback = EvalCallback(
            Monitor(evaluation_environment),
            callback_on_new_best=callback_on_best,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.averaging_window,
        )
        # Set huge number of steps because termination is based on the callback
        agent.learn(int(1e10), callback=eval_callback)

        # Note that rollout metrics are over 100 episodes, and eval metrics use deterministic actions

        # Save the agent
        agent.save(self.model_file)

        # Close the environment
        training_environment.close()
        evaluation_environment.close()

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
        agent: algorithm.OnPolicyAlgorithm = A2C.load(self.model_file)

        # Check hyperparameters
        print(agent.policy)
        print("(n_steps): " + str(agent.n_steps))
        print("(gamma): " + str(agent.gamma))
        print("(ent_coef): " + str(agent.ent_coef))

        # Reset the environment and observe the initial state
        observation, info = environment.reset()

        # Play the game
        terminated: bool = False
        truncated: bool = False
        while not terminated and not truncated:
            # Run the agent
            action, _states = agent.predict(observation, deterministic=True)

            # Perform the action on the environment and observe the new state
            observation, reward, terminated, truncated, info = environment.step(action)

            # Render the step
            environment.render()

        # Close the environment
        environment.close()
