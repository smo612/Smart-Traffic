import numpy as np
from traffic_env import TrafficEnv

class TrafficRLEnv:
    def __init__(self, test_mode=False):
        self.env = TrafficEnv()
        self.state_dim = 4
        self.action_dim = 2
        self.prev_action = 0
        self.same_action_counter = 0
        self.no_pass_counter = 0
        self.test_mode = test_mode

    def reset(self):
        self.env.reset()
        self.prev_action = 0
        self.same_action_counter = 0
        self.no_pass_counter = 0
        return self._get_state()

    def step(self, action):
        passed, avg_wait, imbalance = self.env.step(action)

        ns = self.env.get_queue_length("NS")
        ew = self.env.get_queue_length("EW")
        total_queue = ns + ew
        reward = 0.0

        # ✅ 基本通車獎勵（降低強度）
        reward += passed * 10.0

        # ✅ 平均等待時間 & 不平衡懲罰（加強）
        reward -= avg_wait * 3.0
        reward -= (imbalance ** 2) * 1.0

        # ✅ 長隊伍額外懲罰
        if total_queue > 20:
            reward -= (total_queue - 20) * 5.0

        # ✅ 若長時間不切燈，逐漸懲罰
        if action == self.prev_action:
            self.same_action_counter += 1
            if self.same_action_counter >= 10:
                reward -= (self.same_action_counter - 9) * 10.0
        else:
            self.same_action_counter = 0

        # ✅ 嘗試切燈鼓勵：視 queue 差異大小加分
        if action != self.prev_action:
            reward += abs(ns - ew) * 3.0

        # ✅ 長時間無車通過懲罰
        if passed == 0:
            self.no_pass_counter += 1
            if self.no_pass_counter >= 20:
                reward -= 300.0
        else:
            self.no_pass_counter = 0

        # ✅ 通過效率加分（通過 / 平均等待）
        reward += (passed / (avg_wait + 1)) * 10.0

        # ✅ Clip 避免學習爆炸
        reward = np.clip(reward, -300, 300)

        self.prev_action = action
        next_state = self._get_state()

        if self.test_mode:
            print(f"[TEST LOG] Action={action}, Passed={passed}, Wait={avg_wait:.1f}, Imb={imbalance}, TotalQ={total_queue}")

        return next_state, reward, False

    def _get_state(self):
        ns = self.env.get_queue_length("NS")
        ew = self.env.get_queue_length("EW")
        imbalance = ns - ew
        phase_id = self._get_phase_id(self.env.light_state)

        return np.array([
            ns / 20.0,
            ew / 20.0,
            imbalance / 20.0,
            phase_id / 2.0
        ], dtype=np.float32)

    def _get_phase_id(self, light_state):
        if light_state in ['NS_GREEN', 'NS_YELLOW']:
            return 0
        elif light_state in ['EW_GREEN', 'EW_YELLOW']:
            return 1
        else:
            return 2
