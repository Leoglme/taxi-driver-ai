from env_setup import make_taxi_env
from q_learning_taxi_driver.q_learning_agent import QLearningAgent
from monte_carlo_agent.monte_carlo_agent import MonteCarloAgent


def select_algorithm():
    """
    Affiche un petit menu dans le terminal pour choisir l'algorithme.
    """
    options = {
        "1": "q_learning",
        "2": "monte_carlo"
    }
    print("=== Choix de l'algorithme ===")
    print("1) Q-learning")
    print("2) Monte Carlo")
    choice = None
    while choice not in options:
        choice = input("Entrez 1 ou 2 puis [Entrée] : ").strip()
    return options[choice]


if __name__ == "__main__":
    # 1. Interaction CLI pour choisir l'algo
    algo = select_algorithm()

    # 2. Création de l'environnement pour l'entraînement
    env = make_taxi_env(render_mode=None)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # 3. Instanciation de l'agent choisi
    if algo == "q_learning":
        agent = QLearningAgent(n_states, n_actions)
    else:
        agent = MonteCarloAgent(n_states, n_actions)

    # 4. Entraînement
    agent.train(env, num_episodes=50000, max_steps=200)

    # 5. Boucle de test en mode visuel
    env = make_taxi_env(render_mode="human")
    num_test_episodes = 10
    for episode in range(num_test_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        print(f"\n🔹 Test {episode + 1}")

        # Affichage initial
        env.render()

        while not done:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            # Pompe la fenêtre à chaque étape
            env.render()

        if reward == 20:
            print(f"🏆 Succès en {steps} étapes !")
        else:
            print(f"💀 Échec après {steps} étapes…")

    env.close()
