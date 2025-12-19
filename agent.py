import random
import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_values = [self.get_q_value(state, action) for action in self.actions]
        max_q = max(q_values)
        
        # Handle ties randomly
        best_actions = [action for action, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done=False):
        current_q = self.get_q_value(state, action)
        
        # Max Q for next state
        if done:
            target = reward
        else:
            next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
            target = reward + self.gamma * next_max_q
        
        # Q-Learning update rule
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename="q_table.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No saved Q-table found. Starting fresh.")
