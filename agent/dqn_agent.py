from agent.modeL import Model
from agent.replay_buffer import ReplayBuffer
import torch, random


class DQNAgent:
    def __init__(self, state_dim=11, action_dim=3):
        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100_000, batch_size=64)
        #variables
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.update_freq = 1000  
        self.step_counter = 0    
        #models
        self.model = Model(state_dim, 128, 64, action_dim)
        self.model_target = Model(state_dim, 128, 64, action_dim)
        #optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()
        

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # or self.action_dim - 1 if stored

        state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item() # argmax and return tensor -> integer
    
    def train(self):
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return
        #unizp
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        #convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        #predicted Q-values for actions taken
        q_values = self.model(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q targets
        with torch.no_grad():
            q_next = self.model_target(next_states).max(1)[0]
            q_target = rewards + (1 - dones) * self.gamma * q_next

        #loss and backprop
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        #update target network periodically
        self.step_counter += 1
        if self.step_counter % self.update_freq == 0:
            self.update_target_model()

    def update_target_model(self):
        self.model_target.load_state_dict(self.model.state_dict())






