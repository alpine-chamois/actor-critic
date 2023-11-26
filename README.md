## Advantage Actor-Critic (A2C) Reinforcement Learning (RL) Agent

This A2C RL agent is based on the Asynchronous A2C (A3C) agent in [Deep Reinforcement Learning in Action](https://www.manning.com/books/deep-reinforcement-learning-in-action), but with tuned hyperparameters, and without asynchronous processing.

A2C agents combine a Deep Q-network (DQN) like that used by [DeepMind](https://www.deepmind.com/publications/playing-atari-with-deep-reinforcement-learning) with a policy network like REINFORCE. They provide direct sampling of actions from a distribution (like a policy network) whilst also supporting rapid online learning (like a DQN, but without the need for experience replay or a target network). 

The A2C agent learns to play the [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) game environment in [Gymnasium](https://gymnasium.farama.org/content/basic_usage/):

![Agent-environment loop](images/agent_environment_loop.png)

_OpenAI Gym, OpenAI, 2022_

The agent is a two-headed feed-forward neural network:

![A2C model](images/actor_critic_model.png)

_Deep Reinforcement Learning in Action, Manning, 2020_

Here the agent is being trained to play Cart Pole.

![Training metrics](images/training-metrics.png)

And here the trained agent is playing the game unaided:

![Evaluations](images/evaluation.png)

### What about playing other games?

The A2C agent can be used to play any game as long as the size of the observation space and action space are set in ```actor-critic-agent.py``` and ```main.py``` is updated to correctly handle rewards. There is a [branch of this repository](https://github.com/alpine-chamois/actor-critic/tree/lunar-lander) that shows how it can be successfully trained to play the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) game in Gymnasium.

### Why A2C and not A3C or PPO?

Although A3C and PPO agents can perform better than A2C agents, they include additional complexity that makes the fundamentals of RL more difficult to understand when looking at the code. This A2C agent is designed to be a reference for how to implement a Deep RL (DRL) agent using [PyTorch](https://pytorch.org/). If you want a PPO agent, I recommend using [the implementation in Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). There is a [branch of this repository](https://github.com/alpine-chamois/actor-critic/tree/stable-baselines) that shows how to implement an equivalent A2C agent using Stable Baselines 3, and to convert this agent to a PPO agent, simply replace instances of ```A2C``` with instances of ```PPO```.

### Getting started (Linux)

__Prerequisites:__ Python 3.10 

1. Run the install script:
    ```
    . install.sh
    ```
1. Run the example:
    ```
    (venv) >python -m actorcritic --train --render
    ```
