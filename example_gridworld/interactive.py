from gym3 import Interactive

from example_gridworld.env import GridWorldEnv


def main():
    env = GridWorldEnv()
    ia = Interactive(env, info_key="rgb", synchronous=True)
    ia.run()


if __name__ == "__main__":
    main()
