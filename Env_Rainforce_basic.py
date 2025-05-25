import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Balance_Game_basic import EmotionGameCore

class EmotionBalanceEnv(gym.Env):
    def __init__(self, enabled=False):
        super(EmotionBalanceEnv, self).__init__()
        self.game = EmotionGameCore(hole_enabled=False,
                                    headless=not enabled,
                                    enabled=enabled)

        # 관측: [ball_x, ball_y, bar_left_y, bar_right_y, ball_vx]
        high = np.array([
            self.game.WIDTH,
            self.game.HEIGHT,
            self.game.HEIGHT,
            self.game.HEIGHT,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # 3가지 이산 액션: 왼쪽 바↑, 오른쪽 바↑, 유지
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.prev_y = self.game.ball_y
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
        # 성공 / 실패 보상
        if self.game.success:
            return +100
        if self.game.game_over:
            return -100

        # 1) 높이 비율 보상: 0(bottom) ~ 1(top)
        height_ratio = (self.game.HEIGHT - self.game.ball_y) / self.game.HEIGHT

        # 2) 가로 중앙 비율: 0(edge) ~ 1(center)
        x_dist = abs(self.game.ball_x - self.game.bar_center_x)
        max_dist = self.game.bar_width / 2
        x_ratio = max(0.0, 1.0 - x_dist / max_dist)

        # 3) 위로 움직인 양(ΔY)
        delta_y = self.prev_y - self.game.ball_y

        # 4) 합산 보상
        reward = height_ratio + x_ratio + delta_y

        # 5) 상태 업데이트
        self.prev_y = self.game.ball_y

        return reward

    def render(self):
        self.game.render()
