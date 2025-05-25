import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Balance_Game_basic import EmotionGameCore
import math

class EmotionBalanceEnv(gym.Env):
    def __init__(self, enabled=False):
        super(EmotionBalanceEnv, self).__init__()
        self.game = EmotionGameCore(hole_enabled=False, headless=not enabled, enabled=enabled)

        high = np.array([
            self.game.WIDTH, self.game.HEIGHT, self.game.HEIGHT, self.game.HEIGHT, np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(3)  # 0: 왼쪽 바 올리기, 1: 오른쪽 바 올리기, 2: 유지

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.prev_y = self.game.ball_y  # 초기 prev_y 설정
        return self._get_obs(), {}

    def step(self, action):
        self.game.apply_action(action)
        self.game.update()

        obs = self._get_obs()
        done = self.game.game_over
        reward = self._compute_reward()
        info = {"success": self.game.success}

        return obs, reward, done, False, info

    def _get_obs(self):
        return np.array([
            self.game.ball_x,
            self.game.ball_y,
            self.game.bar_left_y,
            self.game.bar_right_y,
            self.game.ball_vx
        ], dtype=np.float32)

    def _compute_reward(self):
        if self.game.success:
            return +1000
        elif self.game.game_over:
            return -100
        else:
            # 1) 기본 보상 (y축 높이 + 속도 + 근접 보너스)
            height_ratio = (self.game.HEIGHT - self.game.ball_y) / self.game.HEIGHT
            height_reward = 10.0 * height_ratio

            delta_y = self.prev_y - self.game.ball_y
            vel_reward = 2.0 * delta_y

            goal_bonus = 0
            if self.game.ball_y < self.game.HEIGHT * 0.1:
                goal_bonus = 100

            base_reward = height_reward + vel_reward + goal_bonus

            # 2) x축 정렬 계수 (0~1)
            x_dist = abs(self.game.ball_x - self.game.bar_center_x)
            x_ratio = max(0.0, 1.0 - x_dist / (self.game.bar_width / 2))

            # 3) 최종 보상: base_reward를 x_ratio로 스케일링
            reward = base_reward * x_ratio

            # 4) 패널티
            reward += -0.01  # time penalty
            reward += -0.005 * abs(self.game.ball_vx)  # speed penalty

            self.prev_y = self.game.ball_y
            return reward

    def render(self):
        self.game.render()
