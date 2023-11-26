## Advantage Actor-Critic (A2C) Reinforcement Learning (RL) Agent

This [Stable Baselines A2C agent](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) learns to play the [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) game environment in [Gymnasium](https://gymnasium.farama.org/content/basic_usage/):

Here the agent is being trained to play Cart Pole with the same hyperparameters as the PyTorch implementation on [main](https://github.com/alpine-chamois/actor-critic/tree/main).

![Training metrics](images/training-metrics.png)

It learns to play with the same performance in the same number of steps:

![Evaluations](images/evaluation.png)

### Stable Baselines links
* [Reinforcement Learning Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

### Getting started

__Prerequisites:__ Python 3.10 

1. Run the install script (Linux):
    ```
    . install.sh
    ```
    or (Windows):
    ```
    install.bat
    ```
1. Run the example:
    ```
    (venv) >python -m actorcritic --train --render
    ```
1. View the training metrics on tensorboard:
    ```
   (venv) >tensorboard --logdir ./a2c_cartpole_tensorboard/
    ```
