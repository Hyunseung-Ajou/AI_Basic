import pygame
import math
import random

class EmotionGameCore:
    def __init__(self, hole_enabled=True, headless=False, enabled=False):
        pygame.init()
        self.WIDTH, self.HEIGHT = 720, 700
        self.hole_enabled = hole_enabled
        self.headless = headless
        self.enabled = enabled

        if not self.headless and self.enabled:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Emotion Game (RL Core)")
        else:
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))  # 비주얼 비활성화용 Surface

        self.clock = pygame.time.Clock()

        self.bar_center_x = self.WIDTH // 2
        self.bar_width = 300
        self.bar_left_y = self.HEIGHT - 100
        self.bar_right_y = self.HEIGHT - 100
        self.bar_thickness = 6

        self.ball_radius = 18
        self.gravity = 0.6
        self.friction = 0.98

        self.goal_rect = pygame.Rect(self.WIDTH // 2 - 50, 40, 100, 40)

        self.holes = []
        self.reset()

    def generate_holes(self, num_holes=20, margin=10):
        radius = self.ball_radius
        self.holes = []
        attempts = 0
        safe_x_min = self.WIDTH // 2 - 160 + radius + margin
        safe_x_max = self.WIDTH // 2 + 160 - radius - margin
        safe_y_min = 150 + radius + margin
        safe_y_max = self.HEIGHT - 200 - radius - margin

        while len(self.holes) < num_holes and attempts < 1000:
            x = random.randint(safe_x_min, safe_x_max)
            y = random.randint(safe_y_min, safe_y_max)
            overlap = any(math.hypot(x - hx, y - hy) < 2 * radius + margin for hx, hy in self.holes)
            if not overlap:
                self.holes.append((x, y))
            attempts += 1

    def reset(self):
        self.bar_left_y = self.HEIGHT - 100
        self.bar_right_y = self.HEIGHT - 100
        self.ball_x = self.bar_center_x + random.uniform(-10, 10)
        self.ball_y = self.bar_left_y - self.ball_radius - 5
        self.ball_vx = 0
        self.game_over = False
        self.success = False

        if self.hole_enabled:
            self.generate_holes()
        else:
            self.holes = []

    def apply_action(self, action):
        if action == 0:
            self.bar_left_y -= 2
        elif action == 1:
            self.bar_left_y += 2
        elif action == 2:
            self.bar_right_y -= 2
        elif action == 3:
            self.bar_right_y += 2
        elif action == 4:
            pass  # 유지

    def update(self):
        slope = (self.bar_right_y - self.bar_left_y) / self.bar_width
        ball_ax = slope * self.gravity
        self.ball_vx += ball_ax
        self.ball_vx *= self.friction
        self.ball_x += self.ball_vx

        bar_x0 = self.bar_center_x - self.bar_width // 2
        self.ball_y = self.bar_left_y + (self.ball_x - bar_x0) * slope - self.ball_radius - 5

        # === 바에 중력 적용 ===
        gravity_strength = 0.5
        self.bar_left_y += gravity_strength
        self.bar_right_y += gravity_strength

        # 화면 아래로 바가 너무 내려가는 걸 방지
        max_y = self.HEIGHT - 50
        self.bar_left_y = min(self.bar_left_y, max_y)
        self.bar_right_y = min(self.bar_right_y, max_y)

        if self.ball_x < bar_x0 or self.ball_x > self.bar_center_x + self.bar_width // 2:
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
            return  # headless 모드에서는 렌더링 생략

        # 🎯 PyGame 이벤트 처리 (추가)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # 화면 출력 (enabled True일 때만 화면에 보여줌)
        self.screen.fill((30, 30, 30))  # 배경
        pygame.draw.rect(self.screen, (255, 255, 255), self.goal_rect, border_radius=8)
        pygame.draw.line(self.screen, (80, 50, 20),
                         (self.bar_center_x - self.bar_width // 2, self.bar_left_y),
                         (self.bar_center_x + self.bar_width // 2, self.bar_right_y),
                         self.bar_thickness)
        pygame.draw.circle(self.screen, (200, 200, 255),
                           (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        for hx, hy in self.holes:
            pygame.draw.circle(self.screen, (80, 0, 0), (hx, hy), self.ball_radius - 2)
            pygame.draw.circle(self.screen, (255, 50, 50), (hx, hy), self.ball_radius, 2)

        if self.enabled:
            pygame.display.flip()
            self.clock.tick(60)

