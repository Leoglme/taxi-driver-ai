import threading
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import random
from copy import deepcopy
import tkinter as tk
from tkinter import ttk  # Pour les widgets plus modernes si nécessaire
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Définition des emplacements (Rouge, Vert, Jaune, Bleu)
LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 3)]


def manhattan_distance(x1, y1, x2, y2):
    """Calcule la distance de Manhattan entre deux points."""
    return abs(x1 - x2) + abs(y1 - y2)


# === Réseau neuronal avec GRU ===
class RNNNet(nn.Module):
    """
    Réseau de neurones récurrent avec GRU pour la politique de l'agent.
    Prend en entrée l'état du taxi et produit des scores pour chaque action.
    """

    def __init__(self):
        super().__init__()
        # Taille de l'entrée : taxi_row, taxi_col, passenger_pos, destination (4 caractéristiques normalisées)
        self.input_size = 4
        # Couche GRU avec 64 unités cachées, gère les séquences (batch_first=True)
        self.gru = nn.GRU(self.input_size, 64, batch_first=True)
        # Couche entièrement connectée pour mapper les sorties du GRU aux 6 actions possibles
        self.fc = nn.Linear(64, 6)

    def forward(self, x, hidden=None):
        """
        Passe avant du réseau.
        Args:
            x (torch.Tensor): Tensor d'entrée de forme (batch_size, sequence_length, input_size).
            hidden (torch.Tensor, optional): État caché initial du GRU. Par défaut à None.
        Returns:
            tuple: (output, hidden) où output est le score des actions et hidden est le nouvel état caché.
        """
        # Applique la couche GRU
        out, hidden = self.gru(x, hidden)
        # Squeeze la dimension de séquence (elle est de 1 car nous traitons un pas de temps à la fois)
        # et applique la couche entièrement connectée
        return self.fc(out.squeeze(1)), hidden


# === Individu génétique ===
class Individual:
    """
    Représente un individu dans la population de l'algorithme génétique.
    Chaque individu possède un modèle RNNNet et un score de fitness.
    """

    def __init__(self):
        self.model = RNNNet()
        self.fitness = 0

    def clone(self):
        """Crée une copie profonde de l'individu."""
        clone = Individual()
        # Copie l'état du modèle (poids et biais) de l'individu actuel vers le clone
        clone.model.load_state_dict(deepcopy(self.model.state_dict()))
        return clone


def prepare_input(state):
    """
    Prépare l'état de l'environnement pour être utilisé comme entrée du réseau de neurones.
    Normalise les valeurs pour les rendre adaptées à l'entrée du réseau.
    """
    taxi_row, taxi_col, passenger_pos, destination = state
    # Crée un tenseur avec les caractéristiques normalisées
    # passenger_pos est normalisé par 4.0 car il peut être 0, 1, 2, 3 (locations) ou 4 (dans le taxi)
    # destination est normalisé par 3.0 car il peut être 0, 1, 2, 3 (locations)
    return torch.tensor([[
        taxi_row / 4.0,
        taxi_col / 4.0,
        passenger_pos / 4.0,
        destination / 3.0
    ]], dtype=torch.float32).unsqueeze(1)  # Ajoute une dimension pour la séquence (batch_first=True)


def get_target(passenger_pos, destination):
    """
    Détermine la cible actuelle du taxi (passager à récupérer ou destination finale).
    """
    # Si le passager n'est pas encore dans le taxi (passenger_pos < 4), la cible est sa position de départ.
    # Sinon (passenger_pos == 4), la cible est la destination finale.
    return LOCATIONS[destination] if passenger_pos == 4 else LOCATIONS[passenger_pos]


def evaluate(indiv: Individual, env, episodes=5):
    """
    Évalue la fitness d'un individu en faisant jouer son modèle dans l'environnement.
    La fitness est basée sur la récompense totale moyenne sur plusieurs épisodes.
    """
    model = indiv.model
    model.eval()  # Met le modèle en mode évaluation (désactive dropout/batchnorm si présents)
    total_reward = 0

    for _ in range(episodes):
        obs, _ = env.reset()  # Réinitialise l'environnement
        done, hidden = False, None  # Indicateur de fin d'épisode et état caché du GRU
        visited, steps, max_steps = {}, 0, 200  # Suivi des états visités et des pas

        taxi_row, taxi_col, passenger_pos, destination = env.unwrapped.decode(obs)
        target = get_target(passenger_pos, destination)
        prev_dist = manhattan_distance(taxi_row, taxi_col, *target)  # Distance initiale à la cible

        while not done and steps < max_steps:
            input_tensor = prepare_input(env.unwrapped.decode(obs))
            with torch.no_grad():  # Désactive le calcul des gradients pour l'inférence
                output, hidden = model(input_tensor, hidden)
            action = torch.argmax(output).item()  # Sélectionne l'action avec le score le plus élevé

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_taxi_row, next_taxi_col, next_passenger_pos, destination = env.unwrapped.decode(next_obs)

            new_target = get_target(next_passenger_pos, destination)
            new_dist = manhattan_distance(next_taxi_row, next_taxi_col, *new_target)

            # Récompense pour se rapprocher de la cible
            if new_dist < prev_dist:
                reward += 3
            else:
                reward -= 5
            prev_dist = new_dist

            # Bonus/malus pour le ramassage (pickup action == 4)
            if action == 4 and passenger_pos < 4:  # Si l'action est 'pickup' et le passager n'est pas encore pris
                expected_pickup_location = LOCATIONS[passenger_pos]
                if (taxi_row, taxi_col) == expected_pickup_location:
                    reward += 30  # Grande récompense si le pickup est réussi à la bonne position
                else:
                    reward -= 1  # Petit malus si le pickup est tenté au mauvais endroit

            # Pénalité pour rester sur place
            if next_obs == obs:
                reward -= 50

            # Pénalité pour revisiter un état trop souvent (pour éviter les boucles infinies)
            visited[obs] = visited.get(obs, 0) + 1
            if visited[obs] > 3:
                reward -= 100

            total_reward += reward
            obs = next_obs
            taxi_row, taxi_col, passenger_pos, destination = env.unwrapped.decode(obs)
            done = terminated or truncated
            steps += 1

    indiv.fitness = total_reward / episodes  # Moyenne des récompenses sur tous les épisodes


def mutate(indiv: Individual, rate=0.02):
    """
    Applique une mutation aux poids du modèle de l'individu.
    Ajoute un bruit aléatoire aux paramètres du modèle.
    """
    for param in indiv.model.parameters():
        if param.requires_grad:  # S'assure que le paramètre est modifiable
            # Ajoute un bruit gaussien de moyenne 0 et d'écart-type `rate`
            param.data += torch.randn_like(param) * rate


def crossover(p1: Individual, p2: Individual) -> Individual:
    """
    Combine deux individus parents (p1 et p2) pour créer un nouvel individu enfant.
    Les poids de l'enfant sont la moyenne des poids des parents.
    """
    child = Individual()
    # Itère sur les paramètres des modèles des parents et de l'enfant
    for cp, p1p, p2p in zip(child.model.parameters(), p1.model.parameters(), p2.model.parameters()):
        # Les données du paramètre de l'enfant sont la moyenne des données des parents
        cp.data = (p1p.data + p2p.data) / 2
    return child


def genetic_algorithm(env, generations=10, population_size=30, callback=None):
    """
    Implémentation de l'algorithme génétique pour entraîner des agents Taxi.
    """
    population = [Individual() for _ in range(population_size)]
    best_fitness_history = []
    avg_fitness_history = []

    for gen in range(generations):
        # Évalue chaque individu de la population
        for indiv in population:
            evaluate(indiv, env)

        # Trie la population par fitness décroissante
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Enregistre la meilleure fitness et la fitness moyenne pour cette génération
        current_best_fitness = population[0].fitness
        current_mean_fitness = np.mean([i.fitness for i in population])
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(current_mean_fitness)

        if callback:
            # Appelle le callback pour mettre à jour l'interface graphique
            callback(gen, current_best_fitness, current_mean_fitness)

        # Crée la prochaine génération
        next_gen = [population[0].clone(), population[1].clone()]  # Garde les deux meilleurs (élitisme)

        while len(next_gen) < population_size:
            # Sélectionne aléatoirement deux parents parmi les 10 meilleurs de la population
            p1, p2 = random.choices(population[:10], k=2)
            child = crossover(p1, p2)  # Crée un enfant par croisement
            mutate(child)  # Applique une mutation à l'enfant
            next_gen.append(child)  # Ajoute l'enfant à la prochaine génération

        population = next_gen  # La nouvelle génération devient la population actuelle

    # Retourne le meilleur agent trouvé et l'historique des fitness pour le tracé
    return population[0], best_fitness_history, avg_fitness_history


def test_agent(agent: Individual, episodes=3):
    """
    Teste le meilleur agent entraîné en le visualisant dans l'environnement.
    """
    # Crée un environnement avec rendu visuel ('human')
    env = gym.make("Taxi-v3", render_mode="human")
    model = agent.model
    model.eval()  # Passe le modèle en mode évaluation

    for i in range(episodes):
        obs, _ = env.reset()
        done, hidden = False, None
        steps = 0

        print(f"\n--- Début de l'épisode {i + 1} ---")

        # Initialisation pour le suivi de la distance de Manhattan
        taxi_row, taxi_col, passenger_pos, destination = env.unwrapped.decode(obs)
        target = get_target(passenger_pos, destination)
        prev_dist = manhattan_distance(taxi_row, taxi_col, *target)

        while not done and steps < 200:  # Limite à 200 pas par épisode
            input_tensor = prepare_input(env.unwrapped.decode(obs))
            with torch.no_grad():
                output, hidden = model(input_tensor, hidden)
            action = torch.argmax(output).item()  # L'action choisie par le modèle

            # Effectue l'action dans l'environnement
            obs, _, terminated, truncated, _ = env.step(action)
            env.render()  # Affiche l'état actuel de l'environnement

            # Mise à jour de la distance de Manhattan (pour information, non utilisé pour la récompense ici)
            next_taxi_row, next_taxi_col, next_passenger_pos, destination = env.unwrapped.decode(obs)
            new_target = get_target(next_passenger_pos, destination)
            new_dist = manhattan_distance(next_taxi_row, next_taxi_col, *new_target)
            prev_dist = new_dist

            done = terminated or truncated
            steps += 1
        print(f"Épisode {i + 1} terminé en {steps} étapes.")

    env.close()  # Ferme l'environnement de rendu


# === Interface graphique ===
class TaxiApp:
    def __init__(self, master):
        self.master = master
        self.env = gym.make("Taxi-v3", render_mode="ansi")  # Utilise 'ansi' pour l'entraînement
        master.title("Taxi Driver AI - Optimisation par Algorithme Génétique")
        master.geometry("800x700")  # Taille de fenêtre par défaut

        # Frame principale
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Label d'état d'entraînement
        self.label = ttk.Label(main_frame, text="🧠 Entraînement en cours...", font=("Arial", 14))
        self.label.pack(pady=10)

        # Zone de log
        self.log_frame = ttk.LabelFrame(main_frame, text="Journal d'entraînement", padding="10")
        self.log_frame.pack(pady=10, padx=10, fill=tk.X)

        self.log = tk.Text(self.log_frame, height=10, width=70, state='disabled', font=("Courier New", 10))
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.config(yscrollcommand=self.log_scrollbar.set)

        # Zone pour le graphique (initialement vide)
        self.plot_frame = ttk.LabelFrame(main_frame, text="Évolution de la Fitness", padding="10")
        self.plot_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.canvas_plot = None
        self.toolbar_plot = None

        # Lance l'entraînement après un court délai pour que la fenêtre s'affiche
        master.after(100, self.train)

    def log_write(self, txt):
        """Ajoute du texte au widget de log."""
        self.log.config(state='normal')  # Active le widget pour l'écriture
        self.log.insert(tk.END, txt + "\n")
        self.log.see(tk.END)  # Fait défiler jusqu'à la fin
        self.log.config(state='disabled')  # Désactive le widget après écriture
        self.master.update_idletasks()  # Met à jour l'interface immédiatement

    def train(self):
        """Lance le processus d'entraînement de l'algorithme génétique."""

        # Lance l'entraînement dans un thread séparé pour ne pas bloquer l'interface
        def run_ga():
            self.best_agent, best_history, avg_history = genetic_algorithm(
                self.env,
                generations=15,  # Nombre de générations pour l'entraînement
                population_size=30,  # Taille de la population
                callback=lambda g, b, m: self.master.after(0, self.log_write,
                                                           f"Génération {g + 1:2d} | Meilleure: {b:7.2f} | Moyenne: {m:7.2f}")
            )
            self.master.after(0, self.training_finished, best_history, avg_history)

        threading.Thread(target=run_ga).start()

    def training_finished(self, best_history, avg_history):
        """Appelée une fois l'entraînement terminé."""
        self.log_write("✅ Entraînement terminé.")
        self.label.config(text="🎉 Entraînement terminé ! 🎉")

        # Affiche le graphique des résultats
        self.plot_fitness_history(best_history, avg_history)

        # Ajoute un bouton pour lancer le test de l'agent
        self.test_button = ttk.Button(self.master, text="🚀 Tester l'Agent (Visualisation)",
                                      command=self.launch_test_thread)
        self.test_button.pack(pady=10)

    def plot_fitness_history(self, best_history, avg_history):
        """Trace l'historique de la fitness et l'affiche dans l'interface Tkinter."""
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)  # Crée une figure matplotlib

        generations = range(1, len(best_history) + 1)
        ax.plot(generations, best_history, label='Meilleure Fitness', color='green', marker='o', markersize=4)
        ax.plot(generations, avg_history, label='Fitness Moyenne', color='blue', linestyle='--')

        ax.set_title("Évolution de la Fitness au fil des Générations")
        ax.set_xlabel("Générations")
        ax.set_ylabel("Fitness (Récompense Moyenne)")
        ax.legend()
        ax.grid(True)
        ax.set_xticks(generations)  # Assure que chaque génération est marquée sur l'axe X

        # Intègre le graphique dans l'interface Tkinter
        if self.canvas_plot:  # Nettoie les anciens widgets si le graphique est mis à jour
            self.canvas_plot.get_tk_widget().destroy()
            self.toolbar_plot.destroy()

        self.canvas_plot = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas_plot_widget = self.canvas_plot.get_tk_widget()
        self.canvas_plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Ajoute la barre d'outils (zoom, pan, etc.)
        self.toolbar_plot = NavigationToolbar2Tk(self.canvas_plot, self.plot_frame)
        self.toolbar_plot.update()
        self.canvas_plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_plot.draw()  # Dessine le graphique

    def launch_test_thread(self):
        """Lance l'agent de test dans un thread séparé et ferme la fenêtre de l'application."""
        self.master.destroy()  # Ferme la fenêtre Tkinter

        def run():
            test_agent(self.best_agent)
            sys.exit(0)  # Quitte le programme après le test

        threading.Thread(target=run).start()


if __name__ == "__main__":
    # Assurez-vous que l'environnement gym est créé avant de lancer l'application Tkinter
    # pour éviter des problèmes de thread si gym a des initialisations spécifiques.

    root = tk.Tk()
    app = TaxiApp(root)
    root.mainloop()  # Lance la boucle principale de l'interface graphique

