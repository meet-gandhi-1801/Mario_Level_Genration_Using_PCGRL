import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNet.model import CNet
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from train_agent import MarioRepairEnv, get_wrong_tiles

LV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'LevelGenerator', 'RandomDestroyed', 'lv0.txt'
)

print("Loading environment and agent...")
env = MarioRepairEnv(LV_PATH)
model = PPO.load("mario_repair_agent")

obs, _ = env.reset()
wrong_before = len(get_wrong_tiles(env.level))
print(f"Wrong tiles BEFORE: {wrong_before}")

terminated = False
truncated = False
total_reward = 0

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

wrong_after = len(get_wrong_tiles(env.level))
fixed = wrong_before - wrong_after

print(f"Wrong tiles AFTER:  {wrong_after}")
print(f"Fixed:              {fixed} / {wrong_before}  ({100*fixed//wrong_before if wrong_before else 0}%)")
print(f"Total reward:       {total_reward:.1f}")