import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from environment.snake_game import SnakeEnv
from agent.dqn_agent import DQNAgent

CHECKPOINT_PATH = "checkpoints/snake_dqn_ep900.pth"

def main():
    env = SnakeEnv()
    agent = DQNAgent()

    agent.model.load_state_dict(torch.load(CHECKPOINT_PATH))
    agent.model.eval()
    agent.update_target_model()
    agent.epsilon = 0.0

    state = env.reset()
    total_score = 0

    while True:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        env.render()

        state = next_state
        total_score += reward

        if done:
            print(f"Game over! Score: {env.score}")
            break

    env.close()
if __name__ == "__main__":
    main()
