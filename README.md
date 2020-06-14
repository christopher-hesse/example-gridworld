# Example Gridworld

A gym reinforcement learning environment using the [`gym3`](https://github.com/openai/gym3) API.  This is a simple gridworld from Pieter Abbeel's [CS287](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf) to illustrate how to make a simple `gym3` environment.

<img src="https://raw.githubusercontent.com/christopher-hesse/example-gridworld/master/docs/env.gif">

## Installation

```
git clone https://github.com/christopher-hesse/example-gridworld.git
pip install -e example-gridworld
```

## Quick Start

Play the gridworld using a keyboard:

```
python -m example_gridworld.interactive
```

Create the `gym3` environment in code:

```
from example_gridworld import GridWorldEnv
env = GridWorldEnv(num=2)
```

Create a `gym` environment from the `gym3` one:

```
from gym3 import ToGymEnv
from example_gridworld import GridWorldEnv
env = GridWorldEnv(num=1)
gym_env = ToGymEnv(env)
```

## Resources

* [Kenney Micro Roguelike Assets](https://kenney.nl/assets/micro-roguelike)