import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TimeLimit
from Env_Rainforce_basic import EmotionBalanceEnv

# ✅ GPU 확인 출력
if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU 사용 불가 — 현재 CPU 모드로 실행됩니다.")

# ✅ 진행률 출력 콜백
class PrintStepCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 1000 == 0:
            print(f"🔄 Step {self.n_calls} 진행 중...")
        return True

# ✅ 학습 or 플레이 선택
MODE = input("enter the mode (train or play) : ")  # "train" or "play"

if MODE == "train":
    # ✅ 학습 환경 설정 (렌더링 비활성화)
    env = EmotionBalanceEnv(enabled=False)
    check_env(env)

    eval_env = Monitor(TimeLimit(EmotionBalanceEnv(enabled=False), max_episode_steps=300))
    eval_env.render = lambda *args, **kwargs: None

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )


    # ✅ DQN 모델 정의 (DoubleDQN + DuelingDQN)
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.3,
        exploration_fraction=0.5,
        policy_kwargs=dict(net_arch=[128, 128, 64]),
        verbose=1,
        tensorboard_log="./dqn_tensorboard/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ✅ 학습 실행
    model.learn(total_timesteps=1_000_000, callback=[eval_callback, PrintStepCallback()])

    # ✅ 마지막 모델 저장
    model.save("dqn_emotion_balance")
    print("✅ 학습 완료. best_model.zip 및 dqn_emotion_balance.zip 저장됨.")

elif MODE == "play":
    model_path = "./best_model/best_model.zip"
    print(f"✅ 베스트 모델 로드 중: {model_path}")

    # ✅ 플레이용 환경: enabled=True (렌더링), headless=False (창 생성)
    play_env = EmotionBalanceEnv(enabled=True)

    model = DQN.load(model_path)

    obs, _ = play_env.reset()
    done = False
    step_count = 0
    max_steps = 10000  # 안전장치

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = play_env.step(action)
        play_env.render()
        step_count += 1

    print("✅ 플레이 완료.")

