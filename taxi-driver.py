import numpy as np
import gymnasium as gym
import random
from gymnasium.spaces import Discrete

# Création de l'environnement Taxi
env = gym.make("Taxi-v3", render_mode=None)

# Vérification que les espaces sont de type Discrete
assert isinstance(env.observation_space, Discrete), "L'espace d'observation doit être Discrete"
assert isinstance(env.action_space, Discrete), "L'espace d'action doit être Discrete"

# Paramètres du Q-learning
learning_rate = 0.1  # Taux d'apprentissage
discount_factor = 0.99  # Importance des récompenses futures
epsilon = 1.0  # Taux initial d'exploration
epsilon_decay = 0.999  # Décroissance d'epsilon après chaque épisode
min_epsilon = 0.01  # Valeur minimale d'epsilon
num_episodes = 50000  # Nombre d'épisodes d'entraînement
max_steps = 200  # Nombre maximal d'actions par épisode (limite de l'environnement Taxi)

# Initialisation de la Q-table (500 états x 6 actions)
n_states = env.observation_space.n  # 500
n_actions = env.action_space.n  # 6
Q_table = np.zeros((n_states, n_actions))

# Entraînement de l'agent
for episode in range(num_episodes):
    state, _ = env.reset()

    for step in range(max_steps):
        # Sélection d'une action avec la stratégie epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(Q_table[state])  # Exploitation

        new_state, reward, terminated, truncated, info = env.step(action)

        # La récompense de Taxi est déjà définie :
        # -1 par pas, +20 pour un dépôt réussi, -10 pour une action illégale.
        # (On peut éventuellement appliquer du reward shaping supplémentaire ici.)
        td_target = reward + discount_factor * np.max(Q_table[new_state])
        td_error = td_target - Q_table[state, action]
        Q_table[state, action] += learning_rate * td_error

        state = new_state

        if terminated or truncated:
            break

    # Décroissance progressive d'epsilon pour réduire l'exploration au fil de l'entraînement
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 1000 == 0:
        print(f"Épisode {episode}, Epsilon: {epsilon:.3f}")

print("Entraînement terminé !")

# Phase de test (mode "human" pour visualiser)
env = gym.make("Taxi-v3", render_mode="human")
num_test_episodes = 10

for episode in range(num_test_episodes):
    state, _ = env.reset()
    step = 0
    print(f"\n🔹 Test {episode + 1}")

    while True:
        env.render()
        action = np.argmax(Q_table[state])
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
        step += 1

        if terminated or truncated:
            # En cas de dépôt réussi, l'environnement retourne la récompense +20
            if reward == 20:
                print(f"🏆 Succès : Passager déposé avec succès en {step} étapes !")
            else:
                print(f"💀 Échec après {step} étapes...")
            break

env.close()