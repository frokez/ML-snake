import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from environment.snake_game import SnakeEnv
from agent.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

env =  SnakeEnv()
agent = DQNAgent()
scores = []
average_scores = []


NUM_EPISODES = 1000  
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.train()
        

        state = next_state
        total_reward += reward

        if done:
            scores.append(env.score)
            average_score = sum(scores[-100:]) / min(len(scores), 100)
            average_scores.append(average_score)
            print(f"Episode {episode}, Score: {env.score}, Avg(100): {average_score:.2f}, Epsilon: {agent.epsilon:.3f}")
            if agent.epsilon > agent.min_epsilon: # epsilon decay
                agent.epsilon *= agent.epsilon_decay
                agent.epsilon = max(agent.epsilon, agent.min_epsilon)
            break
        if episode % 100 == 0:  #save every 100 episodes
            torch.save(agent.model.state_dict(), f"checkpoints/snake_dqn_ep{episode}.pth")

plt.plot(scores, label='Score')
plt.plot(average_scores, label='Average Score (last 100)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Progress')
plt.legend()
plt.grid()
plt.show()




print(f"Episode {episode}, Score: {env.score}, Epsilon: {agent.epsilon:.3f}")