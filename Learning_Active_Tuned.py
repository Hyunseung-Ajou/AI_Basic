import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from Env_Rainforce_Tuned import EmotionBalanceEnv

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
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.3,
        exploration_fraction=0.7,
        policy_kwargs={
            # Fully-connected 레이어를 3~4개로 늘리고, 폭도 크게
            "net_arch": [512, 512, 256, 128],
            "activation_fn": torch.nn.ReLU,
        },
        verbose=1,
        tensorboard_log="./dqn_tensorboard",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    start_time = time.time()
    model.learn(total_timesteps=1_000_000, callback=[eval_callback, PrintStepCallback()])
    end_time = time.time()

    print(f"⏱️ 학습 소요 시간: {end_time - start_time:.2f}초")

    model_path = "./best_model/best_model.zip"
    model = DQN.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 저장된 Best Model 평가 시작...")
    print(f"불러온 모델 경로: {model_path}")
    n_episodes = 30
    success_count = 0
    rewards = []

    for episode in range(n_episodes):
        play_env = EmotionBalanceEnv(enabled=False)
        obs, _ = play_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 1000

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = play_env.step(action)
            #play_env.render()
            episode_reward += reward
            step_count += 1

        success = info.get("success", False)
        if success:
            success_count += 1
        rewards.append(episode_reward)
        print(f"🎮 Episode {episode + 1}: Reward={episode_reward:.2f}, Success={success}")

    print("\n📊 Best Model 평가 결과")
    if rewards:
        print(f"✅ 성공률: {success_count}/{n_episodes} ({(success_count / n_episodes) * 100:.1f}%)")
        print(f"🏆 평균 보상: {np.mean(rewards):.2f}")
    else:
        print("⚠️ 평균 보상 계산 실패: 보상이 기록되지 않았습니다.")


elif MODE == "play":
    model_path = "./best_model/best_model.zip"
    model = DQN.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    n_episodes = 10
    success_count = 0
    rewards = []

    for episode in range(n_episodes):
        play_env = EmotionBalanceEnv(enabled=True)
        obs, _ = play_env.reset()
        done = False
        step_count = 0
        max_steps = 1000
        episode_reward = 0

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = play_env.step(action)
            play_env.render()
            episode_reward += reward
            step_count += 1

        success = info.get("success", False)
        if success:
            success_count += 1
        rewards.append(episode_reward)
        print(f"🎮 Episode {episode + 1}: Reward={episode_reward:.2f}, Success={success}")

    print(f"✅ 테스트 완료: 성공률 {success_count}/{n_episodes} ({(success_count / n_episodes) * 100:.1f}%)")
    print(f"평균 보상: {np.mean(rewards):.2f}")

else:
    print("⚠️ MODE를 잘못 입력하셨습니다. 'train' 또는 'play'로 입력해주세요.")
