import numpy as np
import gymnasium as gym
import random
from gymnasium.spaces import Discrete

env = gym.make("Taxi-v3", render_mode=None)

assert isinstance(env.observation_space, Discrete)
assert isinstance(env.action_space, Discrete)

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
num_episodes = 50000
max_steps = 200

n_states = env.observation_space.n
n_actions = env.action_space.n
Q_table = np.zeros((n_states, n_actions))

def choose_action(state, Q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state])

for episode in range(num_episodes):
    state, _ = env.reset()
    action = choose_action(state, Q_table, epsilon)

    for step in range(max_steps):
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_action = choose_action(next_state, Q_table, epsilon)

        td_target = reward + discount_factor * Q_table[next_state, next_action]
        td_error = td_target - Q_table[state, action]
        Q_table[state, action] += learning_rate * td_error

        state = next_state
        action = next_action

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 1000 == 0:
        print(f"Épisode {episode}, Epsilon: {epsilon:.3f}")

print("✅ Entraînement SARSA terminé !")

env = gym.make("Taxi-v3", render_mode="human")
num_test_episodes = 10

for episode in range(num_test_episodes):
    state, _ = env.reset()
    print(f"\n🔹 Test SARSA {episode + 1}")
    steps = 0

    while True:
        env.render()
        action = np.argmax(Q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        steps += 1

        if terminated or truncated:
            if reward == 20:
                print(f"🏁 Succès en {steps} étapes")
            else:
                print(f"❌ Échec après {steps} étapes")
            break

env.close()
