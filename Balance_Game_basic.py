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

        # 바 설정
        self.bar_center_x = self.WIDTH // 2
        self.bar_width = 300
        self.bar_left_y = self.HEIGHT - 100
        self.bar_right_y = self.HEIGHT - 100
        self.bar_thickness = 6

        # 공 물리 설정 (난이도 완화)
        self.ball_radius = 18
        self.gravity = 0.3        # 중력 절반으로 감소
        self.friction = 0.995     # 마찰 증가

        # 목표 영역 확대
        self.goal_rect = pygame.Rect(self.WIDTH // 2 - 80, 40, 160, 60)

        self.holes = []
        self.reset()

    def generate_holes(self, num_holes=5, margin=50):
        """구멍 수와 마진을 조정하여 난이도 완화"""
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
            overlap = any(
                math.hypot(x - hx, y - hy) < 2 * radius + margin
                for hx, hy in self.holes
            )
            if not overlap:
                self.holes.append((x, y))
            attempts += 1

    def reset(self):
        """게임 초기화"""
        self.bar_left_y = self.HEIGHT - 100
        self.bar_right_y = self.HEIGHT - 100
        # 공 초기 위치를 중앙 근처로
        self.ball_x = self.bar_center_x + random.uniform(-10, 10)
        self.ball_y = self.bar_left_y - self.ball_radius - 5
        self.ball_vx = 0
        self.game_over = False
        self.success = False

        if self.hole_enabled:
            # 구멍 개수 및 마진 완화
            self.generate_holes(num_holes=5, margin=50)
        else:
            self.holes = []

    def apply_action(self, action):
        """액션 적용: 바를 더 빠르게 이동"""
        speed = 4   # 기존 2 -> 4
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
        """물리 업데이트 및 충돌/성공 판정"""
        # 공 움직임
        slope = (self.bar_right_y - self.bar_left_y) / self.bar_width
        ball_ax = slope * self.gravity
        self.ball_vx += ball_ax
        self.ball_vx *= self.friction
        self.ball_x += self.ball_vx

        # 공 y좌표 계산
        bar_x0 = self.bar_center_x - self.bar_width // 2
        self.ball_y = self.bar_left_y + (self.ball_x - bar_x0) * slope \
                      - self.ball_radius - 5

        # 바에 중력 적용 (천천히 내려감)
        gravity_strength = 0.2   # 기존 0.5 -> 0.2
        self.bar_left_y += gravity_strength
        self.bar_right_y += gravity_strength

        # 바가 화면 아래로 너무 내려가지 않도록 제한
        max_y = self.HEIGHT - 50
        self.bar_left_y = min(self.bar_left_y, max_y)
        self.bar_right_y = min(self.bar_right_y, max_y)

        # 게임 오버/성공 판정
        if self.ball_x < bar_x0 or self.ball_x > bar_x0 + self.bar_width:
            self.game_over = True
        elif self.is_in_hole(self.ball_x, self.ball_y):
            self.game_over = True
        elif self.is_in_goal(self.ball_x, self.ball_y):
            self.success = True
            self.game_over = True

    def is_in_hole(self, x, y):
        """구멍 판정"""
        return any(math.hypot(x - hx, y - hy) < self.ball_radius
                   for hx, hy in self.holes)

    def is_in_goal(self, x, y):
        """골인지점 판정"""
        return self.goal_rect.collidepoint(x, y)

    def render(self):
        """화면 렌더링 (enabled=True일 때만)"""
        if self.headless:
            return

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # 배경
        self.screen.fill((30, 30, 30))
        # 목표
        pygame.draw.rect(self.screen, (255, 255, 255),
                         self.goal_rect, border_radius=8)
        # 바
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
