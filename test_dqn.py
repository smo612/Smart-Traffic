import pygame
import torch
import numpy as np
import argparse
from traffic_env import TrafficEnv
from dqn_agent import DQNAgent
from rl_env import TrafficRLEnv

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="results/best_model.pth", help="Path to model file")
args = parser.parse_args()

pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("DQN Traffic Light Test")
clock = pygame.time.Clock()

env = TrafficRLEnv(test_mode=True)
agent = DQNAgent(env.state_dim, env.action_dim)
agent.policy_net.load_state_dict(torch.load(args.model))
agent.policy_net.eval()
agent.epsilon = 0.0

state = env.reset()
step = 0
max_steps = 1000
running = True
total_passed = 0
total_wait = 0

print(f"✅ 測試模型：{args.model}")

while running and step < max_steps:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = agent.select_action(state)
    next_state, reward, _ = env.step(action)
    state = next_state

    total_passed += env.env.passed_last_step
    total_wait += env.env.total_wait_last_step

    # 顯示當前動作
    print(f"[STEP {step}] Action={action}")

    env.env.draw(screen)
    pygame.display.flip()
    clock.tick(60)
    step += 1

print(f"🚗 測試結束，總通過車輛數：{total_passed}")
print(f"⏱️ 累計等待總時間：{total_wait}")
pygame.quit()
