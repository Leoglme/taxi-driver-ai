import threading
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import random
from copy import deepcopy
import tkinter as tk

LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

# === Réseau neuronal avec GRU ===
class RNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 4
        self.gru = nn.GRU(self.input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 6)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        return self.fc(out.squeeze(1)), hidden

# === Individu génétique ===
class Individual:
    def __init__(self):
        self.model = RNNNet()
        self.fitness = 0

    def clone(self):
        clone = Individual()
        clone.model.load_state_dict(deepcopy(self.model.state_dict()))
        return clone

def prepare_input(state):
    taxi_row, taxi_col, passenger_pos, destination = state
    return torch.tensor([[
        taxi_row / 4.0,
        taxi_col / 4.0,
        passenger_pos / 4.0,
        destination / 3.0
    ]], dtype=torch.float32).unsqueeze(1)

def get_target(passenger_pos, destination):
    return LOCATIONS[destination] if passenger_pos == 4 else LOCATIONS[passenger_pos]

def evaluate(indiv: Individual, env, episodes=5):
    model = indiv.model
    model.eval()
    total_reward = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done, hidden = False, None
        visited, steps, max_steps = {}, 0, 200

        taxi_row, taxi_col, passenger_pos, destination = env.unwrapped.decode(obs)
        target = get_target(passenger_pos, destination)
        prev_dist = manhattan_distance(taxi_row, taxi_col, *target)

        while not done and steps < max_steps:
            input_tensor = prepare_input(env.unwrapped.decode(obs))
            with torch.no_grad():
                output, hidden = model(input_tensor, hidden)
            action = torch.argmax(output).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_taxi_row, next_taxi_col, next_passenger_pos, destination = env.unwrapped.decode(next_obs)

            new_target = get_target(next_passenger_pos, destination)
            new_dist = manhattan_distance(next_taxi_row, next_taxi_col, *new_target)

            if new_dist < prev_dist:
                reward += 3
            else:
                reward -= 5
            prev_dist = new_dist

            # Pickup réussi
            if action == 4 and passenger_pos < 4:
                expected = LOCATIONS[passenger_pos]
                if (taxi_row, taxi_col) == expected:
                    reward += 30
                else:
                    reward -= 1

            if next_obs == obs:
                reward -= 50
            visited[obs] = visited.get(obs, 0) + 1
            if visited[obs] > 3:
                reward -= 100

            total_reward += reward
            obs = next_obs
            taxi_row, taxi_col, passenger_pos, destination = env.unwrapped.decode(obs)
            done = terminated or truncated
            steps += 1

    indiv.fitness = total_reward / episodes

def mutate(indiv: Individual, rate=0.02):
    for param in indiv.model.parameters():
        if param.requires_grad:
            param.data += torch.randn_like(param) * rate

def crossover(p1: Individual, p2: Individual) -> Individual:
    child = Individual()
    for cp, p1p, p2p in zip(child.model.parameters(), p1.model.parameters(), p2.model.parameters()):
        cp.data = (p1p.data + p2p.data) / 2
    return child

def genetic_algorithm(env, generations=10, population_size=30, callback=None):
    population = [Individual() for _ in range(population_size)]
    for gen in range(generations):
        for indiv in population:
            evaluate(indiv, env)
        population.sort(key=lambda x: x.fitness, reverse=True)
        if callback:
            mean = np.mean([i.fitness for i in population])
            callback(gen, population[0].fitness, mean)
        next_gen = [population[0].clone(), population[1].clone()]
        while len(next_gen) < population_size:
            p1, p2 = random.choices(population[:10], k=2)
            child = crossover(p1, p2)
            mutate(child)
            next_gen.append(child)
        population = next_gen
    return population[0]

def test_agent(agent: Individual, episodes=3):
    env = gym.make("Taxi-v3", render_mode="human")
    model = agent.model
    model.eval()

    for _ in range(episodes):
        obs, _ = env.reset()
        done, hidden = False, None
        steps = 0
        taxi_row, taxi_col, passenger_pos, destination = env.unwrapped.decode(obs)
        target = get_target(passenger_pos, destination)
        prev_dist = manhattan_distance(taxi_row, taxi_col, *target)

        while not done and steps < 200:
            input_tensor = prepare_input(env.unwrapped.decode(obs))
            with torch.no_grad():
                output, hidden = model(input_tensor, hidden)
            action = torch.argmax(output).item()
            obs, _, terminated, truncated, _ = env.step(action)
            env.render()
            done = terminated or truncated
            steps += 1
        print(f"Episode terminé en {steps} étapes.")

    env.close()

# === Interface graphique minimaliste ===
class TaxiApp:
    def __init__(self, master):
        self.master = master
        self.env = gym.make("Taxi-v3", render_mode="ansi")
        master.title("Taxi Driver AI - Optimisé")

        self.label = tk.Label(master, text="🧠 Entraînement en cours...")
        self.label.pack(pady=10)

        self.log = tk.Text(master, height=15, width=60)
        self.log.pack(padx=10, pady=10)

        master.after(100, self.train)

    def log_write(self, txt):
        self.log.insert(tk.END, txt + "\n")
        self.log.see(tk.END)
        self.master.update()

    def train(self):
        self.best_agent = genetic_algorithm(
            self.env,
            generations=15,
            population_size=30,
            callback=lambda g, b, m: self.log_write(f"Génération {g} | Best: {b:.1f} | Moyenne: {m:.1f}")
        )
        self.log_write("✅ Entraînement terminé.")
        self.master.after(1000, self.launch_test)

    def launch_test(self):
        self.master.destroy()
        def run():
            test_agent(self.best_agent)
            sys.exit(0)
        threading.Thread(target=run).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = TaxiApp(root)
    root.mainloop()
