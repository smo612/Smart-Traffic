This repo includes the best model trained using the current version of the Dueling Double DQN setup.  
You can run `test_dqn.py` with `results/best_model.pth` to evaluate the policy.  
If you find areas for improvement, feel free to suggest or modify.  
This is a submission-ready version due to the approaching deadline — thanks for understanding.  

### python ver:3.10

```python
pip install torch numpy pygame matplotlib
```

## 📂 File Descriptions

- `train_dqn.py`  
  → Main training script for the Dueling Double DQN agent with reward logging and model saving.

- `test.py`  
  → Standard test script that runs a short evaluation episode and prints performance stats.

- `test.py`  
  → Visualization-only version for infinite loop demo without stats output, suitable for recording.

- `dqn_agent.py`  
  → Implements the Dueling Double DQN agent, including experience replay, soft updates, and epsilon decay.

- `rl_env.py`  
  → Reinforcement learning wrapper around the traffic simulator; defines state, reward, and step logic.

- `traffic_env.py`  
  → Pygame-based simulation of the 4-way intersection with vehicles, phases, and light logic.

- `main.py`  
  → Manual mode to run the traffic environment without any agent (for debugging or animation).

- `config.py`  
  → Centralized configuration for window size, car settings, traffic pattern, phase timing, etc.

- `results/`  
  → Folder to store training results including `best_model.pth`, `reward_curve.png`, and logs.
