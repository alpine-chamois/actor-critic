"""
Main
"""
import argparse
from typing import SupportsFloat

from gymnasium.core import ObsType

from actorcritic.actor_critic_agent import ActorCriticAgent
from actorcritic.agent import Agent


class CartPoleActorCriticAgent(ActorCriticAgent):
    """
    Subclass the agent to shape rewards
    """

    @staticmethod
    def shape_reward(_observation: ObsType, reward: SupportsFloat, terminated: bool,
                     _truncated: bool) -> SupportsFloat:
        """
        Penalise the agent for losing the game
        """
        if terminated:
            reward = -10
        return reward


if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser(description='Actor-Critic')
    parser.add_argument('-t', '--train', action=argparse.BooleanOptionalAction, default=True,
                        help='train the model before evaluating')
    parser.add_argument('-r', '--render', action=argparse.BooleanOptionalAction, default=True,
                        help='render the game during evaluation')
    args: argparse.Namespace = parser.parse_args()

    # Construct agent
    agent: Agent = CartPoleActorCriticAgent(game='CartPole-v1', max_cumulative_reward=500)

    # Train and evaluate
    if args.train is True:
        agent.train()
    agent.evaluate(args.render)
