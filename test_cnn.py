import time
import random
import os

import torch
from sb3_contrib import MaskablePPO
import numpy as np
from snake_game_custom_wrapper_cnn import SnakeEnv
from gym.spaces import Box
from torch.utils.tensorboard import SummaryWriter   # ★ 新增：TensorBoard

experiment_name = "original"
BASE_DIR = os.path.dirname(__file__)

if torch.backends.mps.is_available():
    MODEL_PATH = os.path.join(BASE_DIR, f"trained_models_cnn_mps_{experiment_name}", "ppo_snake_final")
else:
    MODEL_PATH = os.path.join(BASE_DIR, "trained_models_cnn_v2", "ppo_snake_final")

NUM_EPISODE = 100

RENDER = False
FRAME_DELAY = 0.05  # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

# -------- 建一個 dummy env 只拿動作空間 --------
dummy_env = SnakeEnv(seed=0, limit_step=False, silent_mode=True)

# 訓練時經過 VecTransposeImage，給 CNN 的應該是 (3, 84, 84)
chw_observation_space = Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)

custom_objects = {
    # 強制用 CHW 的 obs space，讓 CNN 架構跟訓練時一致
    "observation_space": chw_observation_space,
    "action_space": dummy_env.action_space,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MaskablePPO.load(
    MODEL_PATH,
    custom_objects=custom_objects,  # 用我們覆蓋的 space
    device=device,
)

dummy_env.close()
# ====================================================================

def to_chw(obs: np.ndarray) -> np.ndarray:
    # 如果已經是 CHW 就不用動
    if obs.shape == (3, 84, 84):
        return obs
    # 預設從 HWC -> CHW
    return np.transpose(obs, (2, 0, 1))

# -------- TensorBoard writer（test 用）--------
log_dir = os.path.join(BASE_DIR, "runs", f"test_{experiment_name}")
writer = SummaryWriter(log_dir=log_dir)

total_reward = 0.0
total_score = 0.0
min_score = float("inf")
max_score = float("-inf")

for episode in range(NUM_EPISODE):
    # 每次 episode 用隨機 seed
    seed = random.randint(0, 1_000_000)

    torch.manual_seed(seed)           # PyTorch
    np.random.seed(seed)              # NumPy
    random.seed(seed)                 # Python random

    if RENDER:
        env = SnakeEnv(seed=seed, limit_step=False, silent_mode=False)
    else:
        env = SnakeEnv(seed=seed, limit_step=False, silent_mode=True)
    
    obs = env.reset()
    obs = to_chw(obs)

    episode_reward = 0.0
    done = False
    num_step = 0
    info = None
    sum_step_reward = 0.0

    print(f"=================== Episode {episode + 1} ==================")
    while not done:
        # 這裡一樣用 action_masks
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        num_step += 1
        obs, reward, done, info = env.step(action)
        obs = to_chw(obs)

        if done:
            if info["snake_size"] == env.game.grid_size:
                print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
            else:
                last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        elif info["food_obtained"]:
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            sum_step_reward = 0.0
        else:
            sum_step_reward += reward
            
        episode_reward += reward
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = env.game.score
    min_score = min(min_score, episode_score)
    max_score = max(max_score, episode_score)
    
    snake_size = info["snake_size"] + 1
    print(
        f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, "
        f"Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}"
    )
    total_reward += episode_reward
    total_score += episode_score

    # -------- TensorBoard：每個 episode 的資料 --------
    writer.add_scalar("test/score", episode_score, episode)
    writer.add_scalar("test/reward", episode_reward, episode)
    writer.add_scalar("test/steps", num_step, episode)
    writer.add_scalar("test/snake_size", snake_size, episode)

    env.close() 
    
    if RENDER:
        time.sleep(ROUND_DELAY)

avg_score = total_score / NUM_EPISODE
avg_reward = total_reward / NUM_EPISODE

print(f"=================== Summary ==================")
print(f"Average Score: {avg_score}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {avg_reward}")

# -------- TensorBoard：整體統計 --------
global_step = NUM_EPISODE  # 隨便用一個 step 當作匯總點
writer.add_scalar("test/summary/avg_score", avg_score, global_step)
writer.add_scalar("test/summary/min_score", min_score, global_step)
writer.add_scalar("test/summary/max_score", max_score, global_step)
writer.add_scalar("test/summary/avg_reward", avg_reward, global_step)

writer.close()
