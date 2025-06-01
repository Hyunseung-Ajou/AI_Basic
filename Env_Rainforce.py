import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from Balance_Game import EmotionGameCore

class EmotionBalanceEnv(gym.Env):
    def __init__(self, enabled=False):
        super(EmotionBalanceEnv, self).__init__()
        self.num_holes = 5
        self.game = EmotionGameCore(hole_enabled=True,  # ✅ 구멍 활성화
                                    headless=not enabled,
                                    enabled=enabled, num_holes=5)

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5 + self.num_holes * 2,), dtype=np.float32)

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
            self.game.ball_vx
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

        # 공의 위치
        ball_x, ball_y = self.game.ball_x, self.game.ball_y
        goal_y = self.game.goal_rect.centery
        goal_x = self.game.goal_rect.centerx

        # ΔY (속도)
        delta_y = self.prev_y - ball_y

        # X 중심 거리
        x_dist = abs(ball_x - goal_x)
        max_x_dist = self.game.bar_width / 2
        x_ratio = max(0.0, 1.0 - (x_dist / max_x_dist))

        # 상태별 보상 설계
        if ball_y >= self.game.goal_rect.bottom:
            # 🎯 목표 아래에 있을 때 → 위로 이동 보상
            velocity_reward = 20.0 * delta_y  # 위로 가면 +, 아래로 가면 -
            x_reward = 5.0 * x_ratio  # X 중심 보조 보상 (약하게)
        else:
            # 🎯 목표 이상에 있을 때 → X 중심 보상만, ΔY 보상 제거
            velocity_reward = 0.0
            x_reward = 20.0 * x_ratio  # X 중심에 가까워질수록 보상 ↑

        # 구멍 페널티
        hole_penalty = 0.0
        for hx, hy in self.game.holes:
            dist = math.hypot(ball_x - hx, ball_y - hy)
            safe_dist = self.game.ball_radius * 2
            if dist < safe_dist:
                hole_penalty += -10.0 * (safe_dist - dist) / safe_dist

        # 구멍 통과 보상
        pass_reward = 0.0
        for hx, hy in self.game.holes:
            if self.prev_y > hy and ball_y <= hy:
                pass_reward += +3.0

        # 최종 보상
        reward = velocity_reward + x_reward + hole_penalty + pass_reward

        self.prev_y = ball_y
        return reward

    def render(self):
        self.game.render()
