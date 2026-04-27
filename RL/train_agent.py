import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from utils.visualization import numpy_level
from CNet.model import CNet

# ─── Load CNet ───────────────────────────────────────────────
NET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CNet', 'dict.pkl')
net = torch.load(NET_PATH, weights_only=False).to("cpu")
net.eval()

THRESHOLD = 0.03
NUM_TILE_TYPES = 11


def get_wrong_tiles(level):
    """Return list of (i,j) positions that are illegal according to CNet."""
    wrong = []
    h, w = len(level), len(level[0])
    for i in range(h):
        for j in range(w):
            tile = level[i][j]
            if 6 <= tile <= 9:  # only pipe tiles need checking
                condition = [i]
                for offset in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    ni, nj = i + offset[0], j + offset[1]
                    if 0 <= ni < h and 0 <= nj < w:
                        condition.append(level[ni][nj])
                    else:
                        condition.append(11)
                x = torch.zeros(97)
                x[0] = condition[0]
                for k in range(1, 9):
                    x[k * 12 - 11 + condition[k]] = 1
                pro = F.softmax(net(x), dim=0)
                valid = [t for t in range(11) if pro[t] >= THRESHOLD]
                if tile not in valid:
                    wrong.append((i, j))
    return wrong


# ─── Environment ─────────────────────────────────────────────
class MarioRepairEnv(gym.Env):
    """
    RL Environment for tile-by-tile Mario level repair.

    State:  flat array of the level grid (normalized)
    Action: pick a tile type (0-10) to place at the current wrong tile
    Reward: +10 if tile is fixed, -1 if still wrong, +50 bonus if all tiles fixed
    """

    metadata = {"render_modes": []}

    def __init__(self, level_path):
        super().__init__()
        with open(level_path) as f:
            lv_str = f.read()
        self.original_level = numpy_level(lv_str)
        self.h = len(self.original_level)
        self.w = len(self.original_level[0])

        # action = which tile type to place (0 to 10)
        self.action_space = spaces.Discrete(NUM_TILE_TYPES)

        # observation = flattened level grid
        self.observation_space = spaces.Box(
            low=0, high=11,
            shape=(self.h * self.w,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        import copy
        self.level = copy.deepcopy(self.original_level)
        self.wrong_tiles = get_wrong_tiles(self.level)
        self.current_idx = 0
        return self._get_obs(), {}          # gymnasium returns (obs, info)

    def _get_obs(self):
        flat = []
        for row in self.level:
            flat.extend(row)
        return np.array(flat, dtype=np.float32)

    def step(self, action):
        if self.current_idx >= len(self.wrong_tiles):
            return self._get_obs(), 0.0, True, False, {}

        i, j = self.wrong_tiles[self.current_idx]
        self.level[i][j] = int(action)

        still_wrong = get_wrong_tiles(self.level)
        fixed = (i, j) not in still_wrong
        reward = 10.0 if fixed else -1.0

        self.current_idx += 1
        terminated = self.current_idx >= len(self.wrong_tiles)
        truncated = False

        if terminated and len(still_wrong) == 0:
            reward += 50.0

        # gymnasium step returns (obs, reward, terminated, truncated, info)
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass


# ─── Train ───────────────────────────────────────────────────
if __name__ == '__main__':
    LV_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'LevelGenerator', 'RandomDestroyed', 'lv0.txt'
    )

    print("Setting up environment...")
    env = MarioRepairEnv(LV_PATH)

    print("Training PPO agent...")
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
    )
    model.learn(total_timesteps=50000)

    os.makedirs("RL", exist_ok=True)
    model.save("RL/mario_repair_agent")
    print("Agent saved to RL/mario_repair_agent.zip")