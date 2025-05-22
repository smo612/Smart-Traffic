import pygame
import random
import numpy as np
from config import *

CAR_COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0),
    (255, 255, 0), (255, 165, 0), (128, 0, 128), (0, 255, 255)
]

SAFE_DISTANCE = 10
MAX_GREEN_FRAMES = 200  # 新增最大綠燈持續時間
passed=5
avg_wait=5
imbalance=5
class Car:
    def __init__(self, direction):
        self.direction = direction
        self.in_intersection = False
        self.color = random.choice(CAR_COLORS)
        self.wait_time = 0
        if direction == 'E': self.x, self.y = 0, 280
        elif direction == 'W': self.x, self.y = WINDOW_WIDTH, 320
        elif direction == 'N': self.x, self.y = 280, 0
        elif direction == 'S': self.x, self.y = 320, WINDOW_HEIGHT

    def check_front_clear(self, car_list):
        for other in car_list:
            if other == self:
                continue
            dx, dy = abs(self.x - other.x), abs(self.y - other.y)
            if self.direction == 'E' and 0 < other.x - self.x < CAR_SIZE + SAFE_DISTANCE and dy < CAR_SIZE:
                return False
            elif self.direction == 'W' and 0 < self.x - other.x < CAR_SIZE + SAFE_DISTANCE and dy < CAR_SIZE:
                return False
            elif self.direction == 'N' and 0 < other.y - self.y < CAR_SIZE + SAFE_DISTANCE and dx < CAR_SIZE:
                return False
            elif self.direction == 'S' and 0 < self.y - other.y < CAR_SIZE + SAFE_DISTANCE and dx < CAR_SIZE:
                return False
        return True

    def move(self, light_state, car_list):
        moved = False
        if self.direction in ['E', 'W'] and 250 <= self.x <= 350:
            self.in_intersection = True
        elif self.direction in ['N', 'S'] and 250 <= self.y <= 350:
            self.in_intersection = True

        allow, slow_approach = False, False
        if self.in_intersection:
            allow = True
        else:
            if self.direction == 'E':
                allow = light_state == 'EW_GREEN' or (light_state == 'EW_YELLOW' and self.x > 200)
                slow_approach = self.x + CAR_SPEED < 238
            elif self.direction == 'W':
                allow = light_state == 'EW_GREEN' or (light_state == 'EW_YELLOW' and self.x < 400)
                slow_approach = self.x - CAR_SPEED > 362
            elif self.direction == 'N':
                allow = light_state == 'NS_GREEN' or (light_state == 'NS_YELLOW' and self.y > 200)
                slow_approach = self.y + CAR_SPEED < 238
            elif self.direction == 'S':
                allow = light_state == 'NS_GREEN' or (light_state == 'NS_YELLOW' and self.y < 400)
                slow_approach = self.y - CAR_SPEED > 362

        if (allow or slow_approach) and self.check_front_clear(car_list):
            if self.direction == 'E': self.x += CAR_SPEED
            elif self.direction == 'W': self.x -= CAR_SPEED
            elif self.direction == 'N': self.y += CAR_SPEED
            elif self.direction == 'S': self.y -= CAR_SPEED
            moved = True

        if not moved:
            self.wait_time += 1

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.x, self.y, CAR_SIZE, CAR_SIZE))

class TrafficEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cars = []
        self.phases = ['ALL_RED', 'EW_GREEN', 'EW_YELLOW', 'ALL_RED', 'NS_GREEN', 'NS_YELLOW']
        self.phase_index = 1
        self.light_state = self.phases[self.phase_index]
        self.phase_timer = 0
        self.spawn_prob = 0.03
        self.last_dynamic_update = -30
        self.max_queue_length = {d: random.randint(4, 12) for d in ['E', 'W', 'N', 'S']}
        self.passed_last_step = 0
        self.total_wait_last_step = 0

    def step(self, action):
        # 處理黃燈階段
        if self.light_state in ['NS_YELLOW', 'EW_YELLOW']:
            self.phase_timer += 1
            if self.phase_timer >= YELLOW_DURATION:
                self.phase_index = (self.phase_index + 1) % len(self.phases)
                self.light_state = self.phases[self.phase_index]
                self.phase_timer = 0
            return self._post_step()

        # Agent 主動切燈
        if action == 1 and self.phase_timer >= MIN_GREEN_FRAMES:
            self.phase_index = (self.phase_index + 1) % len(self.phases)
            self.light_state = self.phases[self.phase_index]
            self.phase_timer = 0

        # ✅ 強制最大綠燈時間限制
        elif self.light_state in ['NS_GREEN', 'EW_GREEN'] and self.phase_timer >= MAX_GREEN_FRAMES:
            self.phase_index = (self.phase_index + 1) % len(self.phases)
            self.light_state = self.phases[self.phase_index]
            self.phase_timer = 0
        else:
            self.phase_timer += 1

        return self._post_step()

    def _post_step(self):
        self.update()
        passed = sum(1 for c in self.cars if c.in_intersection)
        avg_wait = np.mean([c.wait_time for c in self.cars]) if self.cars else 0
        imbalance = abs(self.get_queue_length("NS") - self.get_queue_length("EW"))
        self.passed_last_step = passed
        self.total_wait_last_step = avg_wait
        return passed, avg_wait, imbalance

    def force_switch_phase(self):
        self.phase_timer = MIN_GREEN_FRAMES

    def get_queue_length(self, direction):
        if direction == "NS":
            return sum(1 for c in self.cars if c.direction in ["N", "S"])
        elif direction == "EW":
            return sum(1 for c in self.cars if c.direction in ["E", "W"])
        return 0

    def update(self):
        t = pygame.time.get_ticks() // 1000
        if TRAFFIC_PATTERN == 'random':
            self.spawn_prob = random.uniform(0.02, 0.06)
        elif TRAFFIC_PATTERN == 'low':
            self.spawn_prob = 0.01
        elif TRAFFIC_PATTERN == 'medium':
            self.spawn_prob = 0.03
        elif TRAFFIC_PATTERN == 'high':
            self.spawn_prob = 0.06
        elif TRAFFIC_PATTERN == 'realistic':
            if 0 <= (t % 60) < 15:
                self.spawn_prob = 0.06
            elif 15 <= (t % 60) < 30:
                self.spawn_prob = 0.02
            elif 30 <= (t % 60) < 45:
                self.spawn_prob = 0.07
            else:
                self.spawn_prob = 0.01

        if t - self.last_dynamic_update >= 30:
            self.max_queue_length = {d: random.randint(4, 12) for d in ['E', 'W', 'N', 'S']}
            self.last_dynamic_update = t

        for direction in ['E', 'W', 'N', 'S']:
            queue = [c for c in self.cars if c.direction == direction]
            if len(queue) < self.max_queue_length[direction] and random.random() < self.spawn_prob:
                self.cars.append(Car(direction))

        sorted_cars = []
        for d in ['E', 'W', 'N', 'S']:
            same_dir = [c for c in self.cars if c.direction == d]
            same_dir.sort(key=lambda c: c.x if d in ['E', 'N'] else -c.x if d == 'W' else -c.y)
            sorted_cars += same_dir

        for car in sorted_cars:
            car.move(self.light_state, self.cars)

        self.cars = [c for c in self.cars if -50 <= c.x <= WINDOW_WIDTH + 50 and -50 <= c.y <= WINDOW_HEIGHT + 50]

    def draw(self, screen):
        screen.fill((200, 200, 200))
        pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(250, 0, 100, 600))
        pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(0, 250, 600, 100))

        line_color = (255, 255, 255)
        pygame.draw.line(screen, line_color, (240, 250), (240, 350), 2)
        pygame.draw.line(screen, line_color, (360, 250), (360, 350), 2)
        pygame.draw.line(screen, line_color, (250, 240), (350, 240), 2)
        pygame.draw.line(screen, line_color, (250, 360), (350, 360), 2)

        def draw_traffic_light(base_x, base_y, state):
            colors = {'RED': (255, 0, 0), 'YELLOW': (255, 255, 0), 'GREEN': (0, 255, 0), 'OFF': (0, 0, 0)}
            light_colors = {'RED': 'OFF', 'YELLOW': 'OFF', 'GREEN': 'OFF'}
            if state == 'RED': light_colors['RED'] = 'RED'
            elif state == 'YELLOW': light_colors['YELLOW'] = 'YELLOW'
            elif state == 'GREEN': light_colors['GREEN'] = 'GREEN'

            pygame.draw.circle(screen, colors[light_colors['RED']], (base_x, base_y), 10)
            pygame.draw.circle(screen, colors[light_colors['YELLOW']], (base_x + 30, base_y), 10)
            pygame.draw.circle(screen, colors[light_colors['GREEN']], (base_x + 60, base_y), 10)

        if self.light_state == 'EW_GREEN':
            draw_traffic_light(180, 180, 'GREEN')
            draw_traffic_light(380, 400, 'RED')
        elif self.light_state == 'EW_YELLOW':
            draw_traffic_light(180, 180, 'YELLOW')
            draw_traffic_light(380, 400, 'RED')
        elif self.light_state == 'NS_GREEN':
            draw_traffic_light(180, 180, 'RED')
            draw_traffic_light(380, 400, 'GREEN')
        elif self.light_state == 'NS_YELLOW':
            draw_traffic_light(180, 180, 'RED')
            draw_traffic_light(380, 400, 'YELLOW')
        elif self.light_state == 'ALL_RED':
            draw_traffic_light(180, 180, 'RED')
            draw_traffic_light(380, 400, 'RED')

        for car in self.cars:
            car.draw(screen)
