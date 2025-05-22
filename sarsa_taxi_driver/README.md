# 🚖 SARSA Agent pour Taxi-v3 (Gymnasium)

Ce projet implémente un agent utilisant l’algorithme SARSA (State-Action-Reward-State-Action) pour apprendre à résoudre l’environnement Taxi-v3 de Gymnasium. L’objectif est que l’agent apprenne à transporter un passager à sa destination de manière efficace en maximisant la récompense.

## 🧠 Algorithme

SARSA est un algorithme d’apprentissage par renforcement **on-policy**. Il met à jour sa Q-table en fonction de la séquence (état, action, récompense, nouvel état, nouvelle action). Contrairement à Q-learning (off-policy), SARSA prend en compte la politique actuelle de l’agent pour choisir la prochaine action.

**Formule de mise à jour :**  
Q(s, a) ← Q(s, a) + α [r + γ * Q(s', a') - Q(s, a)]  
où :
- `α` : taux d’apprentissage (learning rate)
- `γ` : facteur de discount (discount factor)
- `s` : état courant
- `a` : action courante
- `r` : récompense obtenue
- `s'` : état suivant
- `a'` : action suivante choisie

## ⚙️ Paramètres d'entraînement

- Environnement : `Taxi-v3` (500 états, 6 actions)
- Stratégie d'exploration : epsilon-greedy
- `learning_rate = 0.1`
- `discount_factor = 0.99`
- `epsilon = 1.0` avec `epsilon_decay = 0.999`
- `min_epsilon = 0.01`
- `episodes = 50 000`
- `max_steps = 200` par épisode

## 🏁 Résultats

- L’agent apprend progressivement à atteindre l’objectif.
- Epsilon décroît au fil des épisodes, ce qui permet de passer de l’exploration à l’exploitation.
- L’agent réussit de plus en plus de trajets vers la fin de l’entraînement.
- Le test final avec rendu graphique montre que l’agent accomplit la mission dans la plupart des cas.

## ▶️ Exécution

Assurez-vous d'avoir installé les dépendances :

```bash
pip install gymnasium numpy
```

Puis lancez le script principal :

```bash
python sarsa_taxi.py
```

À la fin de l'entraînement, 10 épisodes de test sont joués en mode `render_mode="human"` pour visualiser les performances de l’agent.

## 🔭 Pistes d'amélioration

- Comparaison avec Q-learning pour analyser les différences de performance.
- Ajout d’un suivi graphique des récompenses par épisode.
- Application à des environnements plus complexes avec Q-networks (DQN).

## 📁 Fichiers

- `sarsa_taxi.py` : Script principal avec entraînement et test de l’agent SARSA
- `README.md` : Présentation du projet

---