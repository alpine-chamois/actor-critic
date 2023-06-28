"""
Main
"""
import argparse

from actorcritic.actor_critic_agent import ActorCriticAgent
from actorcritic.agent import Agent


if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser(description='Actor-Critic')
    parser.add_argument('-t', '--train', action=argparse.BooleanOptionalAction, default=True,
                        help='train the model before evaluating')
    args: argparse.Namespace = parser.parse_args()

    # Construct agent
    agent: Agent = ActorCriticAgent(game='CartPole-v1', max_cumulative_reward=500)

    # Train and evaluate
    if args.train is True:
        agent.train()
    agent.evaluate()
