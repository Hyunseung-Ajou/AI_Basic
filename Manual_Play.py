import pygame
from Balance_Game_Active import EmotionGameCore

def main():
    # 게임 엔진 초기화
    game = EmotionGameCore(hole_enabled=True, headless=False, enabled=True)

    running = True
    clock = pygame.time.Clock()

    # 키 매핑 안내
    print("""
🎮 키보드 조작 안내
- A: 막대 왼쪽 올리기
- Z: 막대 왼쪽 내리기
- K: 막대 오른쪽 올리기
- M: 막대 오른쪽 내리기
- R: 게임 리셋
- ESC: 종료
""")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 키 입력 처리
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            game.apply_action(0)  # 왼쪽 올리기
        if keys[pygame.K_z]:
            game.apply_action(1)  # 왼쪽 내리기
        if keys[pygame.K_k]:
            game.apply_action(2)  # 오른쪽 올리기
        if keys[pygame.K_m]:
            game.apply_action(3)  # 오른쪽 내리기
        if keys[pygame.K_r]:
            print("🔄 게임 리셋!")
            game.reset()
        if keys[pygame.K_ESCAPE]:
            running = False

        # 게임 상태 업데이트 및 렌더링
        game.update()
        game.render()

        # 종료 조건 확인
        if game.success:
            print("🎉 목표 도달 성공!")
        if game.game_over:
            print("💀 게임 오버!")
            # 잠깐 정지 후 리셋
            pygame.time.delay(1000)
            game.reset()

        clock.tick(60)  # FPS

    pygame.quit()

if __name__ == "__main__":
    main()
