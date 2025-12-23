import math

import gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode

        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            self.step_limit = self.grid_size * 4 # More than enough steps to get the food.
        else:
            self.step_limit = 1e9 # Basically no limit.
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs
    
    def step(self, action):
        # 核心遊戲邏輯
        self.done, info = self.game.step(action)  # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        snake_size = info["snake_size"]
        # 進度（0 ~ 1）：從初始長度到填滿整個盤面的比例
        progress = (snake_size - self.init_snake_size) / self.max_growth

        # ================== 1) 蛇填滿整個盤面：勝利 ==================
        if snake_size == self.grid_size:  # Snake fills up the entire board. Game over.
            # 比原版更大的勝利獎勵，偏激進
            reward = self.max_growth * 0.3
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()

            info['episode'] = {
                'r': reward,
                'l': self.reward_step_counter,
            }
            info['game_score'] = self.game.score
            info['is_win'] = True
            return obs, reward, self.done, info

        # ================== 2) 步數上限：卡太久給結束 ==================
        if self.reward_step_counter > self.step_limit:  # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True
            # 適中懲罰，避免無限繞圈
            reward = -3.0
            info['episode'] = {
                'r': reward,
                'l': self.reward_step_counter,
            }
            info['game_score'] = self.game.score
            info['is_win'] = False
            return obs, reward, self.done, info

        # ================== 3) 撞牆 / 撞到自己：死亡 ==================
        if self.done:  # Snake bumps into wall or itself. Episode is over.
            # 死亡懲罰依照進度逐漸變重：
            # 短蛇死比較沒差，長蛇死比較痛，但不會痛到完全不敢冒險
            death_penalty = 2.5 + 3.5 * max(0.0, progress)  # 約 -2.5 ~ -6.0
            reward = -death_penalty

            info['episode'] = {
                'r': reward,
                'l': self.reward_step_counter,
            }
            info['game_score'] = self.game.score
            info['is_win'] = False
            return obs, reward, self.done, info

        # ================== 4) 吃到食物 ==================
        if info["food_obtained"]:
            # 基本吃到食物的獎勵（每一顆都不錯）
            base_reward = 2.0
            # 長蛇吃一顆額外加成：進度越後面，獎勵成長越快
            length_bonus = 4.0 * progress + 3.0 * (progress ** 2)
            reward = base_reward + length_bonus

            # 每吃到一顆就重設步數計數（避免被 step_limit 提早殺掉）
            self.reward_step_counter = 0
        else:
            # ================== 5) 沒死、沒吃到：方向型 shaping ==================
            # 根據蛇頭與食物的距離變化給小獎勵 / 懲罰
            prev_dist = np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"])
            curr_dist = np.linalg.norm(info["snake_head_pos"] - info["food_pos"])
            dist_delta = prev_dist - curr_dist  # >0 表示更靠近食物

            # 調整成中等強度的 shaping，不會壓過吃食物 / 死亡這兩大事件
            shaping_scale = 0.1
            reward = shaping_scale * (dist_delta / self.board_size)

        # 這版不再使用原本的 max_score / min_score 註解值，
        # 因為 reward 範圍會比舊版大得多，是偏激進的設計。

        return obs, reward, self.done, info

    
    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
        
        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Set the food to red
        obs[self.game.food] = [0, 0, 255]

        # Enlarge the observation to 84x84
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)

        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
