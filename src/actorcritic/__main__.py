import argparse

from actorcritic.actor_critic_agent import ActorCriticAgent
from actorcritic.agent import Agent


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args: argparse.Namespace = parser.parse_args()

    # Construct agent
    agent: Agent = ActorCriticAgent(game="CartPole-v1", max_cumulative_reward=500)

    # Train and evaluate
    if args.train:
        agent.train()
    agent.evaluate(args.render)
