from typing import Sequence, Dict, Any, Tuple
import imageio
import os

import gym3
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WALL = 0
PLAYER = 1
SPACE = 2
WIN = 3
LOSE = 4


class GridWorldMDP(gym3.Env):
    """
    Gym-like interface for Markov Decision Processes where the transistion probabilities are known
    and the states are arranged in a grid.

    Subclasses must provide the states/actions/etc that define the MDP
    """

    def __init__(
        self,
        *,
        states: Sequence[Any],
        actions: Sequence[Any],
        state_probs: Dict[Tuple[Any, ...], Dict[Any, float]],
        rewards: Dict[Tuple[Any, ...], float],
        terminal_states: Sequence[Any],
        step_limit: int,
        seed: int,
    ):
        # check validity
        for s in states:
            for a in actions:
                assert (s, a) in state_probs

        for s, a in state_probs.keys():
            assert s in states
            assert a in actions

        for s in terminal_states:
            assert s in states

        for s, a, sp in rewards.keys():
            assert s in states
            assert a in actions
            assert sp in states

        for sp in state_probs.values():
            assert np.allclose(sum(sp.values()), 1.0)

        self.states = states
        self.actions = actions
        self.state_probs = state_probs
        self.rewards = rewards

        ob_space = gym3.types.TensorType(
            eltype=gym3.types.Discrete(len(states)), shape=()
        )
        ac_space = gym3.types.TensorType(
            eltype=gym3.types.Discrete(len(actions)), shape=()
        )
        super().__init__(ob_space=ob_space, ac_space=ac_space, num=1)

        self._step_limit = step_limit
        self._terminal_states = terminal_states
        self._step_count = 0
        self._state = self.states[0]
        self._rng = np.random.RandomState(seed=seed)
        self._last_obs = (0.0, self.states.index(self._state), True)

    def observe(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rew, img, first = self._last_obs
        return (
            np.array([rew], dtype=np.float32),
            np.expand_dims(img, axis=0),
            np.array([first], dtype=np.bool),
        )

    def act(self, ac: np.ndarray) -> None:
        assert ac.shape == (1,)
        action = ac[0]
        self._step_count += 1
        int_act = self.actions[action]
        sp = self.state_probs[(self._state, int_act)]
        rand = self._rng.uniform()
        total_p = 0
        for state, prob in sp.items():
            total_p += prob
            if rand < total_p:  # type: ignore
                reward = self.rewards.get((self._state, int_act, state), 0)
                self._state = state
                break
        else:
            raise Exception("invalid state prob")

        first = False
        if self._state in self._terminal_states or self._step_count > self._step_limit:
            self._step_count = 0
            self._state = self.states[0]
            first = True

        self._last_obs = (reward, self.states.index(self._state), first)

    def get_info(self):
        return [{}]

    def get_transitions(self):
        return [
            dict(
                states=self.states,
                actions=self.actions,
                state_probs=self.state_probs,
                rewards=self.rewards,
            )
        ]

    def get_state(self):
        return [
            dict(
                step_count=self._step_count,
                state=self._state,
                rng_state=self._rng.get_state(),
            )
        ]

    def set_state(self, state):
        s = state[0]
        self._step_count = s["step_count"]
        self._state = s["state"]
        self._rng.set_state(s["rng_state"])


def GridWorldEnv(num=1):
    """
    A small gridworld from
    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf
    """
    envs = []
    for _ in range(num):
        envs.append(SingleGridWorldEnv())
    return gym3.ConcatEnv(envs)


class SingleGridWorldEnv(GridWorldMDP):
    def __init__(self):
        self.size = (4, 3)  # width, height
        actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # dx, dy, origin is top left

        neighbor_actions = {
            (1, 0): [(0, 1), (0, -1)],
            (-1, 0): [(0, 1), (0, -1)],
            (0, 1): [(1, 0), (-1, 0)],
            (0, -1): [(1, 0), (-1, 0)],
        }

        self.walls = [(1, 1)]
        states = []
        state_probs = {}
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                s = (x, y)
                states.append(s)

                for a in actions:
                    probs = {}
                    for actual_dir in actions:
                        if actual_dir == a:
                            p = 0.8
                        elif actual_dir in neighbor_actions[a]:
                            p = 0.1
                        else:
                            p = 0.0
                        nx = x + actual_dir[0]
                        ny = y + actual_dir[1]
                        if nx < 0 or nx >= self.size[0]:
                            nx = x
                        if ny < 0 or ny >= self.size[1]:
                            ny = y
                        next_state = (nx, ny)
                        if next_state in self.walls:
                            next_state = (x, y)
                        if next_state not in probs:
                            probs[next_state] = 0
                        probs[next_state] += p
                    state_probs[(s, a)] = probs

        terminal_state = (-1, -1)
        states.append(terminal_state)

        for s in [terminal_state] + self.walls:
            for a in actions:
                state_probs[(s, a)] = {s: 1.0}

        self.fixed_rewards = {(3, 2): 1, (3, 1): -1}

        rewards = {}
        for (x, y), rew in self.fixed_rewards.items():
            start_state = (x, y)
            for a in actions:
                end_state = (-1, -1)
                rewards[(start_state, a, end_state)] = rew
                state_probs[(start_state, a)] = {(-1, -1): 1.0}

        space = self._load_asset("space")
        self._tile_id_to_asset = {
            SPACE: space,
            PLAYER: self._load_asset("player", space),
            WALL: self._load_asset("wall"),
            LOSE: self._load_asset("lose", space),
            WIN: self._load_asset("win", space),
        }

        super().__init__(
            states=states,
            actions=actions,
            state_probs=state_probs,
            rewards=rewards,
            terminal_states=[terminal_state],
            step_limit=100,
            seed=0,
        )

    def _load_asset(self, name, background=None):
        filepath = os.path.join(SCRIPT_DIR, "assets", f"{name}.png")
        img = imageio.imread(filepath)[:, :, :3]
        if background is not None:
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if (img[y, x] == (34, 35, 35)).all():
                        img[y, x] = background[y, x]
        return img

    def _render_state(self):
        full_tiles = np.zeros((self.size[1] + 2, self.size[0] + 2), dtype=np.uint8)
        tiles = full_tiles[1:-1, 1:-1]
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if self._state == (x, y):
                    tile_id = PLAYER
                elif (x, y) in self.walls:
                    tile_id = WALL
                elif (x, y) in self.fixed_rewards:
                    reward = self.fixed_rewards[(x, y)]
                    if reward > 0:
                        tile_id = WIN
                    else:
                        tile_id = LOSE
                else:
                    tile_id = SPACE
                tiles[y, x] = tile_id

        SIZE = 8
        bitmap = np.zeros(
            (full_tiles.shape[0] * SIZE, full_tiles.shape[1] * SIZE, 3), dtype=np.uint8
        )
        by = 0
        for y in range(full_tiles.shape[1]):
            bx = 0
            for x in range(full_tiles.shape[0]):
                tile_id = full_tiles[x, y]
                asset = self._tile_id_to_asset[tile_id]
                assert asset.shape == (SIZE, SIZE, 3)
                bitmap[bx : bx + SIZE, by : by + SIZE, :] = asset
                bx += SIZE
            by += SIZE
        return bitmap

    def get_info(self):
        return [{"rgb": self._render_state()}]

    def keys_to_act(self, keys_list):
        result = []
        for keys in keys_list:
            act = None
            key_to_act = {"LEFT": 1, "RIGHT": 0, "UP": 3, "DOWN": 2}
            for key in keys:
                if key in key_to_act:
                    act = np.array([key_to_act[key]], dtype=np.uint8)
            result.append(act)
        return result
