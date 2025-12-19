import numpy as np
import pickle
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, actions, state_size=6, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.actions = actions
        self.state_size = state_size
        self.action_size = len(actions)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural Network Weights (Simple MLP: Input -> 64 -> 64 -> Output)
        self.hidden_size = 64
        
        # He initialization
        self.W1 = np.random.randn(self.state_size, self.hidden_size) * np.sqrt(2/self.state_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2/self.hidden_size)
        self.b2 = np.zeros((1, self.hidden_size))
        
        self.W3 = np.random.randn(self.hidden_size, self.action_size) * np.sqrt(2/self.hidden_size)
        self.b3 = np.zeros((1, self.action_size))
        
        # Target Network Weights
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()
        
        self.target_update_counter = 0
        self.target_update_freq = 1000 # Update target every 1000 steps

        self.memory = ReplayBuffer()
        self.batch_size = 32

    def update_target_network(self):
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()
        print("Target Network Updated")

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def forward(self, state):
        # State shape: (batch_size, state_size)
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        
        # Store for visualization (only for single inference usually)
        if state.shape[0] == 1:
            self.last_activations = {
                'input': state[0],
                'hidden1': self.a1[0],
                'hidden2': self.a2[0],
                'output': self.z3[0]
            }
            
        # Linear output for Q-values
        return self.z3

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        
        q_values = self.forward(np.array(state))
        return self.actions[np.argmax(q_values[0])]

    def learn(self, state, action, reward, next_state, done=False):
        # Just store experience
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # 1. Forward Pass (Main Network)
        z1 = np.dot(states, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        q_values = z3
        
        # 2. Compute Targets (Target Network)
        # Target Q = r + gamma * max(Q_target(next_state))
        next_z1 = np.dot(next_states, self.target_W1) + self.target_b1
        next_a1 = self.relu(next_z1)
        next_z2 = np.dot(next_a1, self.target_W2) + self.target_b2
        next_a2 = self.relu(next_z2)
        next_z3 = np.dot(next_a2, self.target_W3) + self.target_b3
        next_q_values = next_z3
        
        targets = q_values.copy()
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q_values[i])
            
            action_idx = int(actions[i])
            targets[i][action_idx] = target
            
        # 3. Backward Pass (MSE Loss)
        delta3 = 2 * (q_values - targets) / self.batch_size
        
        dW3 = np.dot(a2.T, delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = np.dot(delta3, self.W3.T) * self.relu_deriv(z2)
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        delta1 = np.dot(delta2, self.W2.T) * self.relu_deriv(z1)
        dW1 = np.dot(states.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # Gradient Clipping
        for grad in [dW1, db1, dW2, db2, dW3, db3]:
            np.clip(grad, -1, 1, out=grad)
        
        # 4. Update Weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Update Target Network
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.update_target_network()
            self.target_update_counter = 0

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename="dqn_model.pkl"):
        weights = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }
        with open(filename, "wb") as f:
            pickle.dump(weights, f)

    def load(self, filename="dqn_model.pkl"):
        try:
            with open(filename, "rb") as f:
                weights = pickle.load(f)
                self.W1 = weights['W1']
                self.b1 = weights['b1']
                self.W2 = weights['W2']
                self.b2 = weights['b2']
                self.W3 = weights['W3']
                self.b3 = weights['b3']
                print("Loaded DQN model.")
        except FileNotFoundError:
            print("No saved DQN model found. Starting fresh.")
