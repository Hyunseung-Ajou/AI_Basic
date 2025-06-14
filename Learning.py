import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TimeLimit
from Env_Rainforce import EmotionBalanceEnv

if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU 사용 불가 — 현재 CPU 모드로 실행됩니다.")

class PrintStepCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 1000 == 0:
            print(f"🔄 Step {self.n_calls} 진행 중...")
        return True

MODE = input("Enter the mode (train or play) : ").lower()

if MODE == "train":
    # 환경 초기화
    env = Monitor(TimeLimit(EmotionBalanceEnv(enabled=False), max_episode_steps=1000))
    eval_env = Monitor(TimeLimit(EmotionBalanceEnv(enabled=False), max_episode_steps=1000))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

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
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=1,
        tensorboard_log="./dqn_tensorboard/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    model.learn(total_timesteps=100_000, callback=[eval_callback, PrintStepCallback()])
    model.save("dqn_emotion_balance")
    print("✅ 학습 완료. 모델 저장됨 (dqn_emotion_balance.zip)")

elif MODE == "play":
    model_path = "./best_model/best_model.zip"
    print(f"✅ 베스트 모델 로드 중: {model_path}")

    play_env = EmotionBalanceEnv(enabled=True)
    model = DQN.load(model_path, env=play_env, device="cuda" if torch.cuda.is_available() else "cpu")

    obs, _ = play_env.reset()
    done = False
    step_count = 0
    max_steps = 5000

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = play_env.step(action)
        play_env.render()
        step_count += 1

    print("✅ 플레이 완료.")

else:
    print("⚠️ MODE를 잘못 입력하셨습니다. 'train' 또는 'play'로 입력해주세요.")
