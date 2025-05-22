import numpy as np
import random

class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.01
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.Q.shape[1])
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        self.Q[state, action] += self.lr * (td_target - self.Q[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes, max_steps):
        for ep in range(num_episodes):
            state, _ = env.reset()
            for _ in range(max_steps):
                action = self.choose_action(state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                self.update(state, action, reward, new_state)
                state = new_state
                if terminated or truncated:
                    break
            self.decay_epsilon()
            if ep % 1000 == 0:
                print(f"[Q-Learning] Épisode {ep}, ε={self.epsilon:.3f}")
