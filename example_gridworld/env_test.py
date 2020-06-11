import pytest
from gym3 import types_np

from example_gridworld.env import GridWorldEnv

TEST_ENVS = [GridWorldEnv]


@pytest.mark.parametrize("make_env", TEST_ENVS)
def test_works(make_env):
    """
    Make sure the environment works at all and that we can instantiate multiple copies
    """
    envs = []
    for _ in range(3):
        env = make_env()
        envs.append(env)
        for _ in range(10):
            ac = types_np.sample(env.ac_space, bshape=(env.num,))
            env.act(ac)


@pytest.mark.parametrize("make_env", TEST_ENVS)
def test_speed(benchmark, make_env):
    """
    Test the speed of different environments
    """
    env = make_env()
    ac = types_np.zeros(env.ac_space, bshape=(env.num,))

    def loop():
        for _ in range(1000):
            env.act(ac)

    benchmark(loop)
