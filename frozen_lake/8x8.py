import numpy as np
import gymnasium as gym
import random
from gymnasium.spaces import Discrete

# Création de l'environnement FrozenLake pour une map 8x8, déterministe
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode=None)

# Vérification du type des espaces
assert isinstance(env.observation_space, Discrete), "L'espace d'observation doit être Discrete"
assert isinstance(env.action_space, Discrete), "L'espace d'action doit être Discrete"

# Paramètres du Q-learning
learning_rate = 0.1  # Taux d'apprentissage
discount_factor = 0.99  # Importance des récompenses futures
epsilon = 1.0  # Taux initial d'exploration
epsilon_decay = 0.999  # Décroissance progressive d'epsilon
min_epsilon = 0.01  # Valeur minimale d'epsilon
num_episodes = 50000  # Nombre d'épisodes d'entraînement
max_steps = 100  # Nombre maximum d'actions par épisode (adapté pour une map plus grande)

# Initialisation de la Q-table (64 états x 4 actions pour une map 8x8)
n_states = env.observation_space.n  # Pour 8x8, n_states = 64
n_actions = env.action_space.n  # Pour FrozenLake, n_actions = 4
Q_table = np.zeros((n_states, n_actions))

# Entraînement de l'agent avec reward shaping
for episode in range(num_episodes):
    state, _ = env.reset()

    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])

        new_state, reward, terminated, truncated, _ = env.step(action)

        if not (terminated or truncated):
            reward = -0.05  # Pénalité par mouvement
        elif reward == 0:
            reward = -1.0  # Pénalité en cas d'échec (tomber dans un trou)
        elif reward == 1:
            reward = 10.0  # Récompense forte pour atteindre le but

        # Mise à jour de la Q-table (Q-learning classique)
        best_next_action = np.argmax(Q_table[new_state])
        td_target = reward + discount_factor * Q_table[new_state, best_next_action]
        Q_table[state, action] += learning_rate * (td_target - Q_table[state, action])

        state = new_state

        if terminated or truncated:
            break

    # Décroissance d'epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 1000 == 0:
        print(f"Épisode {episode}, Epsilon: {epsilon:.3f}")

print("Entraînement terminé !")

# Phase de test en affichage (mode "human")
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")
num_test_episodes = 10
goal_state = n_states - 1  # Pour une map 8x8, le but est à l'état 63

for episode in range(num_test_episodes):
    state, _ = env.reset()
    step = 0
    print(f"\n🔹 Test {episode + 1}")

    while True:
        env.render()
        action = np.argmax(Q_table[state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        state = new_state
        step += 1

        if terminated or truncated:
            # On vérifie si l'état final est le but (63)
            if state == goal_state or reward == 1.0:
                print(f"🏆 Victoire en {step} étapes !")
            else:
                print(f"💀 Échec après {step} étapes...")
            break

env.close()