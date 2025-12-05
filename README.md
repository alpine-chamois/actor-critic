## Advantage Actor-Critic (A2C) Deep Reinforcement Learning (DRL) Agent

This [Stable Baselines A2C agent](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) learns to play the [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) game environment in [Gymnasium](https://gymnasium.farama.org/content/basic_usage/):

Here the agent is being trained to play Cart Pole with the same hyperparameters as the PyTorch implementation on [main](https://github.com/alpine-chamois/actor-critic/tree/main).

![Training metrics](images/training-metrics.png)

And here the trained agent is playing the game unaided:

![Evaluations](images/evaluation.png)

### Stable Baselines links
* [Reinforcement Learning Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

### Getting started

## Setup
[Install UV](https://docs.astral.sh/uv/getting-started/installation/), then run the following command:

```
uv sync
```

## Run the Example
```
uv run -m  actorcritic --train --render
tensorboard --logdir logs
```

