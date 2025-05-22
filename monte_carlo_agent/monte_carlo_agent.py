import numpy as np
import random
from collections import defaultdict


class MonteCarloAgent:
    def __init__(
            self,
            n_states,
            n_actions,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.999,
            min_epsilon=0.01
    ):
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.returns_sum = defaultdict(lambda: np.zeros(n_actions))
        self.returns_count = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(self.Q[state]))
        return int(np.argmax(self.Q[state]))

    def generate_episode(self, env, max_steps):
        episode = []
        state, _ = env.reset()
        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if terminated or truncated:
                break
        return episode

    def update(self, episode):
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited:
                self.returns_sum[state][action] += G
                self.returns_count[state][action] += 1
                self.Q[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]
                visited.add((state, action))

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes, max_steps):
        for ep in range(num_episodes):
            episode = self.generate_episode(env, max_steps)
            self.update(episode)
            self.decay_epsilon()
            if ep % 1000 == 0:
                print(f"[MonteCarlo] Épisode {ep}, ε={self.epsilon:.3f}")
