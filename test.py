import pygame
import torch
import numpy as np
from dqn_agent import DQNAgent
from rl_env import TrafficRLEnv

# Initialize Pygame window
pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Smart Traffic Light - Demo Mode")
clock = pygame.time.Clock()

# Initialize environment and agent
# test_mode=False disables console logging from rl_env.py
env = TrafficRLEnv(test_mode=False)
agent = DQNAgent(env.state_dim, env.action_dim)
agent.policy_net.load_state_dict(torch.load("results/best_model.pth"))
agent.policy_net.eval()
agent.epsilon = 0.0  # Pure greedy action for evaluation

# Reset environment to start state
state = env.reset()
running = True

print("✅ Demo mode started. Press [X] to exit the window.")

# Main infinite loop (run until manually closed)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Agent selects action and applies to environment
    action = agent.select_action(state)
    next_state, _, _ = env.step(action)
    state = next_state

    # Render updated state
    env.env.draw(screen)
    pygame.display.flip()
    clock.tick(60)  # Limit to 60 FPS
