import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from Balance_Game_Active import EmotionGameCore

class EmotionBalanceEnv(gym.Env):
    def __init__(self, enabled=False):
        super(EmotionBalanceEnv, self).__init__()
        self.num_holes = 8
        self.game = EmotionGameCore(hole_enabled=True,  # ✅ 구멍 활성화
                                    headless=not enabled,
                                    enabled=enabled, num_holes=8)

        high = np.array([
            self.game.WIDTH,
            self.game.HEIGHT,
            self.game.HEIGHT,
            self.game.HEIGHT,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        hole_high = np.tile(
            np.array([self.game.WIDTH, self.game.HEIGHT], dtype=np.float32),
            self.num_holes
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5 + self.num_holes * 2+2,), dtype=np.float32)

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
        obs = [
            self.game.ball_x,
            self.game.ball_y,
            self.game.bar_left_y,
            self.game.bar_right_y,
            self.game.ball_vx,
            self.game.wind_strength * self.game.wind_direction,  # 바람 세기와 방향
            self.game.wind_variation  # 바람 노이즈 크기
        ]
        for (hx, hy) in self.game.holes:
            obs.append(hx)
            obs.append(hy)

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        if self.game.success:
            return +100.0
        if self.game.game_over:
            return -100.0

        ball_x, ball_y = self.game.ball_x, self.game.ball_y
        goal_x = self.game.goal_rect.centerx

        delta_y = self.prev_y - ball_y
        vertical_reward = 2.0 * delta_y

        x_dist = abs(ball_x - goal_x)
        max_x_dist = self.game.bar_width / 2
        x_ratio = max(0.0, 1.0 - (x_dist / max_x_dist))

        height_weight = min(1.0,
                            (ball_y - self.game.goal_rect.bottom) / (self.game.HEIGHT - self.game.goal_rect.bottom))
        x_reward = 3.0 * x_ratio * height_weight

        # 구멍 회피 보상 강화
        hole_penalty = 0.0
        for hx, hy in self.game.holes:
            dist = math.hypot(ball_x - hx, ball_y - hy)
            safe_dist = self.game.ball_radius * 2.5  # 안전 거리 확대
            if dist < safe_dist:
                hole_penalty += -20.0 * (safe_dist - dist) / safe_dist  # 강력한 페널티

        pass_reward = 0.0
        for hx, hy in self.game.holes:
            if self.prev_y > hy and ball_y <= hy:
                pass_reward += +15.0  # 통과 보상 강화

        time_penalty = -0.02

        reward = vertical_reward + x_reward + hole_penalty + pass_reward + time_penalty
        self.prev_y = ball_y
        return reward

    def render(self):
        self.game.render()
