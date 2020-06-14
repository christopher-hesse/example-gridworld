from gym3 import types_np, ViewerWrapper
# from gym3 import VideoRecorderWrapper
from example_gridworld import GridWorldEnv

env = GridWorldEnv(num=1)
env = ViewerWrapper(env, info_key="rgb", tps=3, width=1024, height=768)
# env = VideoRecorderWrapper(env, directory=".", info_key="rgb", fps=3)
step = 0
while True:
    ac = types_np.sample(env.ac_space, bshape=(env.num,))
    env.act(ac)
    rew, obs, first = env.observe()
    print(f"step {step} obs {obs} reward {rew} first {first}")
    if step > 0 and first[0]:
        break
    step += 1
