## Advantage Actor-Critic (A2C) Reinforcement Learning (RL) Agent

This [Stable Baselines A2C agent](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) learns to play the [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) game environment in [Gymnasium](https://gymnasium.farama.org/content/basic_usage/):

Here the agent is being trained to play Cart Pole with the same hyperparameters as the PyTorch implementation on [main](https://github.com/alpine-chamois/actor-critic/tree/main).

![Training metrics](images/training-metrics.png)

_Generated after training using TensorBoard:_ ```tensorboard --logdir ./a2c_cartpole_tensorboard/```

It learns to play with the same performance in the same number of steps:

![Evaluations](images/evaluation.png)

### Stable Baselines links
[Reinforcement Learning Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

### Getting started (Windows)

__Prerequisites:__ Python 3.10 

1. Create a virtual environment:
    ```
    >python -m venv venv
    >venv\Scripts\activate
    ```
1. Install wheel:
    ```
    (venv) >pip install wheel
    ```
1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
1. Install the local package in editable mode:
    ```
    (venv) >pip install -e .\src
    ```
1. Run the example:
    ```
    (venv) >python -m actorcritic --train
    ```
