## Advantage Actor-Critic (A2C) Reinforcement Learning (RL) Agent

This A2C RL agent is based on the DA2C agent in [Deep Reinforcement Learning in Action](https://www.manning.com/books/deep-reinforcement-learning-in-action), but without distributed processing.

A2C agents combine a Deep Q-network (DQN) like that used by [DeepMind](https://www.deepmind.com/publications/playing-atari-with-deep-reinforcement-learning) with a policy network like REINFORCE. They provide direct sampling of actions from a distribution (like a policy network) whilst also supporting rapid online learning (like a DQN, but without the need for experience replay or a target network). 

The A2C agent learns to play the [Cart Pole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) game environment in [OpenAI Gym](https://www.gymlibrary.dev/content/basic_usage/):

![Agent-environment loop](images/agent_environment_loop.png)

_OpenAI Gym, OpenAI, 2022_

The agent is a two-headed feed-forward neural network:

![A2C model](images/actor_critic_model.png)

_Deep Reinforcement Learning in Action, Manning, 2020_

Here the agent is being trained to play Cart Pole until it completes the game.

![Training metrics](images/training-metrics.png)

And here the trained agent is playing the game unaided with 10% observation noise:

![Evaluations](images/evaluation.png)

### Useful Machine Learning Mastery links:
[Choosing the number of hidden layers and neurons in a neural network](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/)

[Choosing activation functions for a neural network](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

[Choosing the learning rate for a neural network](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)
