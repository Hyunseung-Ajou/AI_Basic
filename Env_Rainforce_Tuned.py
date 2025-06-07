import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from Balance_Game_Tuned import EmotionGameCore

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
            return +200.0
        if self.game.game_over:
            return -100.0

        ball_x, ball_y = self.game.ball_x, self.game.ball_y
        goal_x = self.game.goal_rect.centerx
        goal_top = self.game.goal_rect.top

        # x 중심 보상
        x_dist = abs(ball_x - goal_x)
        max_x_dist = self.game.bar_width / 2
        x_center_score = max(0.0, 1.0 - (x_dist / max_x_dist))
        x_weight = 2.5

        # y 상승 보상 (상승만 확실히)
        delta_y = self.prev_y - ball_y
        if delta_y > 0:
            vertical_score = delta_y * 2.0
        else:
            vertical_score = delta_y * 0.2

        # 구멍/바람 근처 판단
        wind_strength = self.game.wind_strength
        wind_threshold = 0.04
        min_hole_dist = min(
            [math.hypot(ball_x - hx, ball_y - hy) for hx, hy in self.game.holes]) if self.game.holes else 999
        danger_zone = self.game.ball_radius * 2.5
        hole_near = min_hole_dist < danger_zone * 2
        strong_wind = wind_strength > wind_threshold

        # y_weight는 항상 1.0 이상, 구멍/바람 근처에도 최소치 보장
        y_weight = 1.2
        if hole_near or strong_wind:
            y_weight = 1.0

        # x 중심이 심하게 깨지면 y 보상을 0으로 두지만, 0.5보다 낮을 때만!
        if x_center_score < 0.5:
            vertical_reward = 0.0
        else:
            vertical_reward = y_weight * vertical_score

        very_close_danger_zone = self.game.ball_radius * 1.1
        very_close_to_hole = False
        for hx, hy in self.game.holes:
            dist = math.hypot(ball_x - hx, ball_y - hy)
            # 아직 홀 아래쪽이면서(지나치지 않았고) 아주 가까울 때만!
            if dist < very_close_danger_zone and ball_y > hy:
                very_close_to_hole = True
                break

        # 기존 y_weight, vertical_reward 코드 이후에 다음을 추가:
        if very_close_to_hole:
            vertical_reward = 0.2  # 아주 가까우면 y보상 wnfdla!

        center_reward = x_weight * x_center_score

        # 목표 근처 추가 shaping
        penalty_coeff = -150.0  # 기본값
        bonus = 0.0
        if delta_y > 0 and ball_y < goal_top + 200:
            bonus = 15.0
            penalty_coeff *= 3.0  # 보너스가 15일 때 패널티도 3배 강화
        elif delta_y > 0 and ball_y < goal_top + 500:
            bonus = 5.0
            penalty_coeff *= 2.0  # 보너스가 5일 때 패널티도 1.5배 강화

        # 구멍 패널티
        hole_penalty = 0.0
        for hx, hy in self.game.holes:
            dist = math.hypot(ball_x - hx, ball_y - hy)
            safe_dist = self.game.ball_radius * 2.0
            if dist < safe_dist:
                #hole_penalty += penalty_coeff  * (safe_dist - dist) / safe_dist
                ratio = (safe_dist - min_hole_dist) / safe_dist  # 0~1
                hole_penalty = penalty_coeff * (ratio ** 2)
                #hole_penalty += -60 * (safe_dist - dist) / safe_dist

        # 구멍 통과 보상
        pass_reward = 0.0
        for hx, hy in self.game.holes:
            if self.prev_y > hy and ball_y <= hy:
                pass_reward += +25.0

        # 시간 패널티
        time_penalty = -0.02

        # 골인 높이에 도달한 경우 x좌표가 골인지점 x에 가까울수록 추가 보상
        x_goal_bonus = 0.0
        if ball_y <= goal_top + 30:
            x_goal_bonus = 2.0 * x_center_score

        reward = center_reward + vertical_reward + hole_penalty + pass_reward + time_penalty + bonus + x_goal_bonus
        self.prev_y = ball_y
        return reward

    def render(self):
        self.game.render()
