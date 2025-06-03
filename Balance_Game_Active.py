import pygame
import math
import random

class EmotionGameCore:
    def __init__(self, hole_enabled=True, headless=False, enabled=False, num_holes=8):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1200, 1000
        self.hole_enabled = hole_enabled
        self.headless = headless
        self.enabled = enabled
        self.num_holes = num_holes

        # 바람 설정
        self.wind_enabled = True  # 바람 활성화 여부
        self.wind_strength = 0.1
        self.wind_direction = 1
        self.wind_update_interval = 100
        self.wind_variation = 0.02
        self.frame_count = 0

        if not self.headless and self.enabled:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Emotion Game (RL Core)")
        else:
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.clock = pygame.time.Clock()

        # 폰트 초기화 (매 프레임 생성 방지)
        self.font = pygame.font.SysFont(None, 24)

        # 막대 설정
        self.bar_center_x = self.WIDTH // 2
        self.bar_width = 450
        self.bar_left_y = self.HEIGHT - 100
        self.bar_right_y = self.HEIGHT - 100
        self.bar_thickness = 6

        # 공 물리 설정
        self.ball_radius = 18
        self.gravity = 0.3
        self.friction = 0.995

        # 목표 영역
        self.goal_rect = pygame.Rect(self.WIDTH // 2 - 125, 40, 250, 80)
        self.reset()

    def reset(self):
        self.bar_left_y = self.HEIGHT - 100
        self.bar_right_y = self.HEIGHT - 100
        self.ball_x = self.bar_center_x + random.uniform(-10, 10)
        self.ball_y = self.bar_left_y - self.ball_radius - 5
        self.ball_vx = 0
        self.game_over = False
        self.success = False

        # 구멍 랜덤 생성
        self.holes = [
            (600 + random.randint(0, 100), 200),
            (500 + random.randint(0, 100), 450),
            (350 + random.randint(0, 100), 600),
            (300 + random.randint(0, 100), 480),
            (730 + random.randint(0, 100), 550),
            (650 + random.randint(0, 100), 700),
            (500 + random.randint(0, 100), 800),
            (400 + random.randint(0, 100), 350)
        ] if self.hole_enabled else []

    def apply_action(self, action):
        speed = 4
        if action == 0:
            self.bar_left_y -= speed
        elif action == 1:
            self.bar_left_y += speed
        elif action == 2:
            self.bar_right_y -= speed
        elif action == 3:
            self.bar_right_y += speed

    def update(self):
        self.frame_count += 1

        # 바람 갱신 (주기적으로)
        if self.wind_enabled and self.frame_count % self.wind_update_interval == 0:
            self.wind_strength = random.uniform(0.02, 0.1)
            self.wind_direction = random.choice([-1, 1])
            self.wind_variation = random.uniform(0.005, 0.02)
            print(f"🌬️ 바람 갱신 - 방향: {'→' if self.wind_direction > 0 else '←'}, 세기: {self.wind_strength:.3f}")

        # 바람 영향 계산
        wind_force = 0.0
        if self.wind_enabled:
            wind_force = self.wind_strength * self.wind_direction
            wind_force += random.uniform(-self.wind_variation, self.wind_variation)

        # 공의 움직임
        slope = (self.bar_right_y - self.bar_left_y) / self.bar_width
        ball_ax = slope * self.gravity
        self.ball_vx += ball_ax + 0.5 * wind_force
        self.ball_vx *= self.friction
        self.ball_x += self.ball_vx

        # 공의 y좌표
        bar_x0 = self.bar_center_x - self.bar_width // 2
        self.ball_y = self.bar_left_y + (self.ball_x - bar_x0) * slope - self.ball_radius - 5

        # 막대 중력
        gravity_strength = 0.2
        self.bar_left_y += gravity_strength
        self.bar_right_y += gravity_strength

        max_y = self.HEIGHT - 30
        min_y = self.goal_rect.bottom + 20
        self.bar_left_y = max(min_y, min(self.bar_left_y, max_y))
        self.bar_right_y = max(min_y, min(self.bar_right_y, max_y))

        # 게임 판정
        if self.ball_x < bar_x0 or self.ball_x > bar_x0 + self.bar_width:
            self.game_over = True
        elif self.is_in_hole(self.ball_x, self.ball_y):
            self.game_over = True
        elif self.is_in_goal(self.ball_x, self.ball_y):
            self.success = True
            self.game_over = True

    def is_in_hole(self, x, y):
        return any(math.hypot(x - hx, y - hy) < self.ball_radius for hx, hy in self.holes)

    def is_in_goal(self, x, y):
        return self.goal_rect.collidepoint(x, y)

    def render(self):
        if self.headless:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((30, 30, 30))

        # 목표 지점
        pygame.draw.rect(self.screen, (255, 255, 255),
                         self.goal_rect, border_radius=8)

        # 막대
        pygame.draw.line(self.screen, (80, 50, 20),
                         (self.bar_center_x - self.bar_width // 2, self.bar_left_y),
                         (self.bar_center_x + self.bar_width // 2, self.bar_right_y),
                         self.bar_thickness)

        # 공
        pygame.draw.circle(self.screen, (200, 200, 255),
                           (int(self.ball_x), int(self.ball_y)), self.ball_radius)

        # 구멍
        for hx, hy in self.holes:
            pygame.draw.circle(self.screen, (80, 0, 0),
                               (hx, hy), self.ball_radius - 2)
            pygame.draw.circle(self.screen, (255, 50, 50),
                               (hx, hy), self.ball_radius, 2)

        # 바람 시각화
        if self.wind_enabled:
            wind_display_x = self.WIDTH // 2
            wind_display_y = 30
            arrow_length = int(self.wind_strength * 500)
            arrow_dx = arrow_length * self.wind_direction

            pygame.draw.line(self.screen, (200, 200, 0),
                             (wind_display_x, wind_display_y),
                             (wind_display_x + arrow_dx, wind_display_y), 4)

            if self.wind_direction > 0:
                pygame.draw.polygon(self.screen, (200, 200, 0), [
                    (wind_display_x + arrow_dx, wind_display_y),
                    (wind_display_x + arrow_dx - 10, wind_display_y - 5),
                    (wind_display_x + arrow_dx - 10, wind_display_y + 5),
                ])
            else:
                pygame.draw.polygon(self.screen, (200, 200, 0), [
                    (wind_display_x + arrow_dx, wind_display_y),
                    (wind_display_x + arrow_dx + 10, wind_display_y - 5),
                    (wind_display_x + arrow_dx + 10, wind_display_y + 5),
                ])

            wind_text = self.font.render(f"Wind: {self.wind_strength * self.wind_direction:+.2f}", True, (255, 255, 255))
            self.screen.blit(wind_text, (wind_display_x - 60, wind_display_y - 20))

        if self.enabled:
            pygame.display.flip()
            self.clock.tick(60)
