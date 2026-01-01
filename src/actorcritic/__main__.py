import argparse
from typing import SupportsFloat, override

from gymnasium.core import ObsType

from actorcritic.actor_critic_agent import ActorCriticAgent
from actorcritic.agent import Agent


class CartPoleActorCriticAgent(ActorCriticAgent):
    """
    Subclass the agent to shape rewards
    """

    @override
    def _shape_reward(
        self,
        _observation: ObsType,  # type: ignore
        reward: SupportsFloat,
        terminated: bool,
        _truncated: bool,
    ) -> SupportsFloat:
        """
        Penalise the agent for losing the game
        """
        if terminated:
            reward = -10
        return reward


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args: argparse.Namespace = parser.parse_args()

    # Construct agent
    agent: Agent = CartPoleActorCriticAgent(game="CartPole-v1", max_cumulative_reward=500)

    # Train and evaluate
    if args.train:
        agent.train()
    agent.evaluate(args.render)
