import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn import SnakeEnv

experiment_name = "v2"

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 8
LOG_DIR = f"logs/PPO_CNN_{experiment_name}"

os.makedirs(LOG_DIR, exist_ok=True)
class SnakeMetricsCallback(BaseCallback):
    """
    TensorBoard記錄性能
    """
    def __init__(self, verbose=0):
        super(SnakeMetricsCallback, self).__init__(verbose)

        self.episode_game_scores = []
        self.episode_wins = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                
                if self.episode_count <= 5:
                    print(f"\n=== Episode {self.episode_count} ===")
                    print(f"Info keys: {list(info.keys())}")
                    if "game_score" in info:
                        print(f"  game_score: {info['game_score']}")
                    if "is_win" in info:
                        print(f"  is_win: {info['is_win']}")
                
                if "game_score" in info:
                    self.episode_game_scores.append(info["game_score"])
                if "is_win" in info:
                    self.episode_wins.append(1 if info["is_win"] else 0)
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_game_scores:
            self.logger.record("game_metrics/mean_game_score", np.mean(self.episode_game_scores))
            self.logger.record("game_metrics/max_game_score", np.max(self.episode_game_scores))
            self.logger.record("game_metrics/min_game_score", np.min(self.episode_game_scores))
            
            self.episode_game_scores = []
        
        if self.episode_wins:
            self.logger.record("game_metrics/win_rate", np.mean(self.episode_wins) * 100)
            self.logger.record("game_metrics/total_wins", np.sum(self.episode_wins))
            self.episode_wins = []

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(seed=0):
    def _init():
        env = SnakeEnv(seed=seed)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def main():

    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = DummyVecEnv([make_env(seed=s) for s in seed_set])

    if torch.backends.mps.is_available():
        lr_schedule = linear_schedule(5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using MPS (Metal Performance Shaders).
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="mps",
            verbose=1,
            n_steps=2048,
            batch_size=512*8,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )
    else:
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using CUDA.
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            n_steps=512,
            batch_size=256,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )

    RESUME_FROM_CHECKPOINT = False  # 從 checkpoint 繼續
    CHECKPOINT_PATH = "trained_models_cnn_original/ppo_snake_45500000_steps.zip"

    if RESUME_FROM_CHECKPOINT:
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        model = MaskablePPO.load(
            CHECKPOINT_PATH,
            env=env,
            device="cuda",
            tensorboard_log=LOG_DIR
        )
        model.batch_size = 1024
        model.n_steps = 2048
        print("Checkpoint loaded successfully!")

    # Set the save directory
    if torch.backends.mps.is_available():
        save_dir = f"trained_models_cnn_mps_{experiment_name}"
    else:
        save_dir = f"trained_models_cnn_{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625 // 2 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        metrics_callback = SnakeMetricsCallback()

        model.learn(
            total_timesteps=int(50000000),
            callback=[checkpoint_callback, metrics_callback],
            # reset_num_timesteps=False #checkpoint callback will reset the num_timesteps
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))

if __name__ == "__main__":
    main()
