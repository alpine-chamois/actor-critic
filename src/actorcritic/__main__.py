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
    parser.add_argument('-r', '--render', action=argparse.BooleanOptionalAction, default=True,
                        help='render the game during evaluation')
    args: argparse.Namespace = parser.parse_args()

    # Construct agent
    agent: Agent = ActorCriticAgent(game='LunarLander-v2', max_cumulative_reward=200)

    # Train and evaluate
    if args.train is True:
        agent.train()
    agent.evaluate(args.render)
