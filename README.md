## Advantage Actor-Critic (A2C) Deep Reinforcement Learning (DRL) Agent

This A2C agent learns to play the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) game environment in [Gymnasium](https://gymnasium.farama.org/content/basic_usage/):

Here the agent is being trained to play Lunar Lander.

![Training metrics](images/training-metrics.png)

And here the trained agent is playing the game unaided:

![Evaluations](images/evaluation.png)

See [main](https://github.com/alpine-chamois/actor-critic/tree/main) for more details.

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
