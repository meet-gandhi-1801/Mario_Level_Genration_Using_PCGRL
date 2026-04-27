"""
Mario Level Generation via Reinforcement Learning
==================================================
The agent generates a Mario level from scratch, tile by tile.
Left to right, top to bottom. Rewarded for valid, playable structure.

Tile legend (from the repo):
 0 = empty (air)
 1 = ground/solid block
 2 = breakable brick
 3 = question mark block
 4 = used block
 5 = platform/floating block
 6 = top of pipe (left)
 7 = top of pipe (right)
 8 = body of pipe (left)
 9 = body of pipe (right)
10 = enemy (goomba/koopa)
11 = border/wall (not placeable)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Tile constants ────────────────────────────────────────────
AIR        = 0
GROUND     = 1
BRICK      = 2
QBLOCK     = 3
USED       = 4
PLATFORM   = 5
PIPE_TL    = 6   # pipe top-left
PIPE_TR    = 7   # pipe top-right
PIPE_BL    = 8   # pipe body-left
PIPE_BR    = 9   # pipe body-right
ENEMY      = 10
NUM_TILES  = 11

# Level dimensions
LEVEL_H = 16   # rows
LEVEL_W = 50   # columns (keepable in ~10 seconds of training)
GROUND_ROW = LEVEL_H - 1   # bottom row = row 15

# ── CNet for structural validation ───────────────────────────
NET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'CNet', 'dict.pkl'
)
try:
    cnet = torch.load(NET_PATH, weights_only=False).to("cpu")
    cnet.eval()
    USE_CNET = True
    print("CNet loaded successfully.")
except Exception as e:
    USE_CNET = False
    print(f"CNet not loaded ({e}), using rule-based rewards only.")


def cnet_tile_valid(level, i, j):
    """Returns True if tile at (i,j) is valid according to CNet."""
    if not USE_CNET:
        return True
    tile = level[i][j]
    if not (6 <= tile <= 9):
        return True   # CNet only checks pipe tiles
    h, w = len(level), len(level[0])
    condition = [i]
    for offset in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        ni, nj = i + offset[0], j + offset[1]
        condition.append(level[ni][nj] if 0 <= ni < h and 0 <= nj < w else 11)
    x = torch.zeros(97)
    x[0] = condition[0]
    for k in range(1, 9):
        x[k * 12 - 11 + condition[k]] = 1
    with torch.no_grad():
        pro = F.softmax(cnet(x), dim=0)
    valid = [t for t in range(11) if pro[t] >= 0.03]
    return tile in valid


# ── Reward helpers ────────────────────────────────────────────

def reward_for_tile(level, i, j):
    """
    Compute immediate reward when agent places tile at (i, j).
    Called right after placement so we can check context.
    """
    tile = level[i][j]
    reward = 0.0

    # ── Ground rules ──
    if i == GROUND_ROW:
        if tile == GROUND:
            reward += 1.0          # ground row should be solid
        else:
            reward -= 2.0          # punish non-ground on bottom row

    # ── Air in sky ──
    if i < GROUND_ROW - 4 and tile == AIR:
        reward += 0.1              # sky should mostly be air

    # ── Enemies should stand on ground, not float ──
    if tile == ENEMY:
        below = i + 1
        if below < LEVEL_H and level[below][j] in [GROUND, BRICK, PLATFORM]:
            reward += 1.0          # enemy on solid surface
        else:
            reward -= 2.0          # floating enemy

    # ── Pipe structure rules ──
    if tile in [PIPE_TL, PIPE_TR]:
        # Pipe top must have pipe body below it
        below = i + 1
        if below < LEVEL_H:
            expected_below = PIPE_BL if tile == PIPE_TL else PIPE_BR
            if level[below][j] == expected_below:
                reward += 2.0
            elif level[below][j] != AIR:
                reward -= 1.0
        # Pipe top-left must have pipe top-right beside it
        if tile == PIPE_TL and j + 1 < LEVEL_W:
            if level[i][j + 1] == PIPE_TR:
                reward += 2.0
            elif level[i][j + 1] != AIR:
                reward -= 1.0

    if tile in [PIPE_BL, PIPE_BR]:
        # Pipe body must have pipe top or more body above it
        above = i - 1
        if above >= 0:
            expected_above = PIPE_TL if tile == PIPE_BL else PIPE_TR
            if level[above][j] in [expected_above, PIPE_BL if tile == PIPE_BL else PIPE_BR]:
                reward += 1.5
            else:
                reward -= 1.0
        # Pipe body must stand on ground
        if i == GROUND_ROW - 1 and level[GROUND_ROW][j] == GROUND:
            reward += 1.0

    # ── Question blocks / bricks should be elevated ──
    if tile in [BRICK, QBLOCK]:
        if 3 <= i <= GROUND_ROW - 4:
            reward += 0.5
        elif i == GROUND_ROW:
            reward -= 0.5          # q-block on floor looks wrong

    # ── CNet structural check ──
    if USE_CNET and not cnet_tile_valid(level, i, j):
        reward -= 3.0

    return reward


def final_level_reward(level):
    """
    Reward computed once at the end of episode for overall level quality.
    """
    h, w = len(level), len(level[0])
    reward = 0.0

    # Ground continuity — penalise gaps
    gap_penalty = 0
    for j in range(w):
        if level[GROUND_ROW][j] != GROUND:
            gap_penalty += 1
    reward -= gap_penalty * 0.5

    # Variety bonus — reward using different tile types
    unique_tiles = len(set(level[i][j] for i in range(h) for j in range(w)))
    reward += unique_tiles * 1.0

    # Playability: at least 60% of bottom row must be ground
    ground_count = sum(1 for j in range(w) if level[GROUND_ROW][j] == GROUND)
    if ground_count / w >= 0.6:
        reward += 10.0
    else:
        reward -= 10.0

    # Penalise too many enemies
    enemy_count = sum(1 for i in range(h) for j in range(w) if level[i][j] == ENEMY)
    if enemy_count > w * 0.1:
        reward -= enemy_count * 0.5

    return reward


# ── Environment ───────────────────────────────────────────────
class MarioGenEnv(gym.Env):
    """
    RL Environment for generating Mario levels tile by tile.

    - Agent fills the grid left-to-right, top-to-bottom
    - State: current partial level (flattened) + current position
    - Action: which tile type to place (0-10)
    - Reward: structural validity + playability rules
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.h = LEVEL_H
        self.w = LEVEL_W
        self.total_tiles = self.h * self.w

        # action = tile type to place
        self.action_space = spaces.Discrete(NUM_TILES)

        # observation = flattened level + normalised position
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.total_tiles + 2,),   # +2 for (row, col) position
            dtype=np.float32
        )
        self.level = None
        self.pos = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.level = [[AIR] * self.w for _ in range(self.h)]
        self.pos = 0
        return self._get_obs(), {}

    def _get_obs(self):
        flat = []
        for row in self.level:
            for tile in row:
                flat.append(tile / NUM_TILES)     # normalise to [0,1]
        row_norm = (self.pos // self.w) / self.h
        col_norm = (self.pos  % self.w) / self.w
        flat.append(row_norm)
        flat.append(col_norm)
        return np.array(flat, dtype=np.float32)

    def step(self, action):
        i = self.pos // self.w
        j = self.pos  % self.w

        # Place the tile
        self.level[i][j] = int(action)

        # Immediate reward for this tile
        reward = reward_for_tile(self.level, i, j)

        self.pos += 1
        terminated = self.pos >= self.total_tiles
        truncated  = False

        if terminated:
            reward += final_level_reward(self.level)

        return self._get_obs(), reward, terminated, truncated, {}

    def get_level(self):
        return [row[:] for row in self.level]


# ── Visualiser ────────────────────────────────────────────────
TILE_COLORS = {
    AIR:      "#87CEEB",   # sky blue
    GROUND:   "#8B4513",   # brown
    BRICK:    "#CD853F",   # brick orange
    QBLOCK:   "#FFD700",   # gold
    USED:     "#A9A9A9",   # grey
    PLATFORM: "#DEB887",   # tan
    PIPE_TL:  "#228B22",   # dark green
    PIPE_TR:  "#228B22",
    PIPE_BL:  "#32CD32",   # green
    PIPE_BR:  "#32CD32",
    ENEMY:    "#FF4500",   # red-orange
}

TILE_LABELS = {
    AIR: "Air", GROUND: "Ground", BRICK: "Brick", QBLOCK: "?Block",
    USED: "Used", PLATFORM: "Platform", PIPE_TL: "Pipe", PIPE_TR: "Pipe",
    PIPE_BL: "Pipe", PIPE_BR: "Pipe", ENEMY: "Enemy"
}

def visualise_level(level, title="Generated Mario Level", save_path=None):
    h, w = len(level), len(level[0])
    fig, ax = plt.subplots(figsize=(w * 0.3, h * 0.3 + 1.5))
    for i in range(h):
        for j in range(w):
            tile = level[i][j]
            color = TILE_COLORS.get(tile, "#FFFFFF")
            rect = plt.Rectangle([j, h - i - 1], 1, 1,
                                  facecolor=color, edgecolor="#333333", linewidth=0.3)
            ax.add_patch(rect)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    # legend
    seen = set()
    patches = []
    for i in range(h):
        for j in range(w):
            t = level[i][j]
            if t not in seen:
                seen.add(t)
                patches.append(mpatches.Patch(color=TILE_COLORS[t], label=TILE_LABELS[t]))
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              bbox_to_anchor=(1.0, 1.0), framealpha=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Level image saved to {save_path}")
    plt.show()


# ── Training callback ─────────────────────────────────────────
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward = 0.0

    def _on_step(self):
        self._ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        return True

    def plot(self):
        if not self.episode_rewards:
            return
        window = 50
        smoothed = np.convolve(
            self.episode_rewards,
            np.ones(window) / window, mode="valid"
        )
        plt.figure(figsize=(10, 4))
        plt.plot(self.episode_rewards, alpha=0.3, label="Episode reward")
        plt.plot(range(window - 1, len(self.episode_rewards)), smoothed,
                 label=f"Smoothed (window={window})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("RL Agent Training Progress — Mario Level Generation")
        plt.legend()
        plt.tight_layout()
        os.makedirs("RL", exist_ok=True)
        plt.savefig("RL/training_curve.png", dpi=150)
        print("Training curve saved to RL/training_curve.png")
        plt.show()


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Mario Level Generation — Full RL Training")
    print("=" * 55)
    print(f"  Level size : {LEVEL_H} x {LEVEL_W} tiles")
    print(f"  Tile types : {NUM_TILES}")
    print(f"  CNet active: {USE_CNET}")
    print("=" * 55)

    env = MarioGenEnv()
    callback = RewardLoggerCallback()

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    print("\nStarting training — this will take 15-30 minutes...")
    print("Watch the 'ep_rew_mean' value — it should rise over time.\n")

    model.learn(total_timesteps=500000, callback=callback)

    os.makedirs("RL", exist_ok=True)
    model.save("RL/mario_gen_agent")
    print("\nAgent saved to RL/mario_gen_agent.zip")

    # Plot training curve
    callback.plot()

    # Generate 5 sample levels and visualise
    print("\nGenerating 5 sample levels with trained agent...")
    for idx in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        level = env.get_level()
        save_path = f"RL/generated_level_{idx+1}.png"
        visualise_level(level, title=f"RL Generated Level #{idx+1}", save_path=save_path)

    print("\nDone! Check the RL/ folder for:")
    print("  mario_gen_agent.zip     — trained model")
    print("  training_curve.png      — reward over time")
    print("  generated_level_*.png   — 5 generated levels")