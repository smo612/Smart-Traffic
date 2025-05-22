# config.py

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
CAR_SIZE = 10
CAR_SPEED = 2
PHASE_TIME = 60  # one phase lasts this many frames
TRAFFIC_PATTERN = 'realistic'  # options: random, low, medium, high, realistic
MIN_GREEN_FRAMES = 60  # 每個相位最短維持秒數（供 RL 強制切燈用）
YELLOW_DURATION = 25

