import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from rl_env import TrafficRLEnv
from config import TRAFFIC_PATTERN
import sys

# ✅ 訓練參數調整
EPISODES = 30000
TEST_INTERVAL = 500

# ✅ 輸出目前 traffic pattern
print(f"🚦 使用交通模式：{TRAFFIC_PATTERN}")

env = TrafficRLEnv()
state_dim = env.state_dim
action_dim = env.action_dim
agent = DQNAgent(state_dim, action_dim)

train_rewards = []
test_rewards = []
best_reward = -float("inf")

print("🧠 開始訓練 DQN 模型... (Ctrl+C 可中斷並自動儲存與畫圖)")

try:
    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for _ in range(300):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # ✅ 強化通車獎勵、加入 reward clip
            if reward != 0:
                reward = np.clip(reward, -1000, 1000)

            agent.store((state, action, reward, next_state, done))
            agent.train_step()
            agent.soft_update()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        train_rewards.append(total_reward)

        print(f"[Train] Ep {ep}, Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

        if ep % TEST_INTERVAL == 0:
            test_env = TrafficRLEnv(test_mode=True)
            state = test_env.reset()
            test_total_reward = 0

            for _ in range(300):
                action = agent.select_action(state)
                next_state, reward, _ = test_env.step(action)
                test_total_reward += reward
                state = next_state

            test_rewards.append(test_total_reward)
            print(f"[Test] Ep {ep} | Reward = {test_total_reward:.2f}")

            if test_total_reward > best_reward:
                best_reward = test_total_reward
                torch.save(agent.policy_net.state_dict(), "results/best_model.pth")
                print(f"✅ Best model updated at Ep {ep} | Reward = {test_total_reward:.2f}")

except KeyboardInterrupt:
    print("🛑 手動中斷訓練，儲存成果中...")

finally:
    # 儲存模型與訓練資料
    torch.save(agent.policy_net.state_dict(), "results/last_model.pth")
    np.save("results/train_rewards.npy", np.array(train_rewards))
    np.save("results/test_rewards.npy", np.array(test_rewards))

    # 畫 reward 曲線圖
    plt.figure()
    plt.plot(train_rewards, label="Train Reward")
    plt.plot(range(0, len(test_rewards) * TEST_INTERVAL, TEST_INTERVAL), test_rewards, label="Test Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training vs Test Reward")
    plt.legend()
    plt.savefig("results/reward_curve.png")
    plt.close()

    print("✅ 已儲存 last_model.pth 與 reward_curve.png")
