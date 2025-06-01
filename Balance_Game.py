import pygame
import math
import random

class EmotionGameCore:
    def __init__(self, hole_enabled=True, headless=False, enabled=False, num_holes=5):
        pygame.init()
        self.WIDTH, self.HEIGHT = 720, 700
        self.hole_enabled = hole_enabled
        self.headless = headless
        self.enabled = enabled
        self.num_holes = num_holes

        # 구멍 위치 고정 (5개)
        self.holes = [
            (350, 200),
            (230, 450),
            (450, 300),
            (400, 480),
            (280, 350)
        ] if self.hole_enabled else []

        if not self.headless and self.enabled:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Emotion Game (RL Core)")
        else:
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.clock = pygame.time.Clock()

        # 막대 설정 (좌우 높이 다르게 움직여 기울기 조절)
        self.bar_center_x = self.WIDTH // 2
        self.bar_width = 300
        self.bar_left_y = self.HEIGHT - 100
        self.bar_right_y = self.HEIGHT - 100
        self.bar_thickness = 6

        # 공 물리 설정
        self.ball_radius = 18
        self.gravity = 0.3
        self.friction = 0.995

        # 목표 영역
        self.goal_rect = pygame.Rect(self.WIDTH // 2 - 80, 40, 160, 60)
        self.reset()

    def reset(self):
        self.bar_left_y = self.HEIGHT - 100
        self.bar_right_y = self.HEIGHT - 100
        self.ball_x = self.bar_center_x + random.uniform(-10, 10)
        self.ball_y = self.bar_left_y - self.ball_radius - 5
        self.ball_vx = 0
        self.game_over = False
        self.success = False


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
        # action == 4: 유지

    def update(self):
        # 공의 움직임
        slope = (self.bar_right_y - self.bar_left_y) / self.bar_width
        ball_ax = slope * self.gravity
        self.ball_vx += ball_ax
        self.ball_vx *= self.friction
        self.ball_x += self.ball_vx

        # 공의 y좌표는 막대의 기울기에 따라 결정 (공은 항상 막대 위에 있음)
        bar_x0 = self.bar_center_x - self.bar_width // 2
        self.ball_y = self.bar_left_y + (self.ball_x - bar_x0) * slope \
                      - self.ball_radius - 5

        # 바에 중력 적용 (서서히 내려감)
        gravity_strength = 0.2
        self.bar_left_y += gravity_strength
        self.bar_right_y += gravity_strength

        # 막대가 화면 아래로 너무 내려가지 않도록 제한
        max_y = self.HEIGHT - 50
        min_y = self.goal_rect.bottom + 20
        self.bar_left_y = max(min_y, min(self.bar_left_y, max_y))
        self.bar_right_y = max(min_y, min(self.bar_right_y, max_y))

        # 게임 오버/성공 판정
        if self.ball_x < bar_x0 or self.ball_x > bar_x0 + self.bar_width:
            self.game_over = True
        elif self.is_in_hole(self.ball_x, self.ball_y):
            self.game_over = True
        elif self.is_in_goal(self.ball_x, self.ball_y):
            self.success = True
            self.game_over = True

    def is_in_hole(self, x, y):
        return any(math.hypot(x - hx, y - hy) < self.ball_radius
                   for hx, hy in self.holes)

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

        if self.enabled:
            pygame.display.flip()
            self.clock.tick(60)
