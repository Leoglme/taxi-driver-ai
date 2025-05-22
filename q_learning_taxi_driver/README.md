# 🧠 Q-Learning Agent pour Taxi-v3 🚕  

Ce module implémente un **agent basé sur l’algorithme de Q-Learning** pour résoudre l’environnement `Taxi-v3` de Gymnasium.

## 📚 Description de l’algorithme

Le Q-Learning est un algorithme **model-free off-policy** d’apprentissage par renforcement. Il permet à un agent d’apprendre à partir de son environnement en mettant à jour une **Q-table** (table des valeurs d’action) à chaque interaction.

Le Q-Learning apprend au fur et à mesure, étape par étape.
À chaque action, il met à jour la Q-table immédiatement en estimant la meilleure récompense future possible.


### 🔁 Logique de mise à jour :
À chaque étape, on met à jour la Q-table selon la formule :

```

Q(s, a) ← Q(s, a) + α × \[r + γ × max(Q(s′, a′)) − Q(s, a)]

```

- `s` : état courant  
- `a` : action choisie  
- `r` : récompense reçue  
- `s′` : nouvel état  
- `α` : taux d’apprentissage (learning rate)  
- `γ` : facteur de discount  
- `max(Q(s′, a′))` : meilleure action possible dans le nouvel état

---

## ⚙️ Paramètres principaux

| Paramètre         | Valeur par défaut | Description                                      |
|-------------------|-------------------|--------------------------------------------------|
| `learning_rate`   | `0.1`             | Vitesse d’apprentissage de l’agent               |
| `discount_factor` | `0.99`            | Pondération des récompenses futures              |
| `epsilon`         | `1.0`             | Probabilité de choisir une action aléatoire      |
| `epsilon_decay`   | `0.999`           | Réduction progressive de l’exploration           |
| `min_epsilon`     | `0.01`            | Valeur minimale d’epsilon                        |
| `num_episodes`    | `50000`           | Nombre d’épisodes pour l’entraînement            |
| `max_steps`       | `200`             | Nombre maximal d’actions par épisode             |

---

## 🧪 Phases de l’entraînement

- **Exploration** : au début, l’agent explore l’environnement avec un taux élevé d’epsilon.
- **Exploitation** : au fur et à mesure que l’agent apprend, `epsilon` diminue, ce qui favorise les actions les plus prometteuses.
- **Stabilisation** : une fois que `epsilon` atteint son minimum, l’agent applique une politique quasi-déterministe basée sur la Q-table.

---

## ✅ Résultats attendus

- Un agent **naïf (ε=1)** peut nécessiter jusqu’à **300–350 étapes** pour terminer un épisode.
- Un agent **entraîné** (avec epsilon réduit et Q-table bien remplie) peut atteindre des **performances optimales autour de 20 étapes** pour réussir.

---

## 📈 Suivi de l'apprentissage

Un log est affiché tous les 1000 épisodes :

```

\[Q-Learning] Épisode 3000, ε=0.406

````

Cela permet de suivre la **décroissance de l'exploration** et de diagnostiquer un éventuel blocage.

---

## 📂 Fichiers

- `q_learning_agent.py` : contient la classe `QLearningAgent` avec toute la logique.
- Utilisé dans `main.py` pour lancer l'entraînement ou les tests.

---

## 💬 Exemple d’utilisation

```bash
$ python main.py
=== Choix de l'algorithme ===
1) Q-learning
2) Monte Carlo
Entrez 1 ou 2 puis [Entrée] : 1
````

Puis l’agent s’entraîne pendant 50 000 épisodes avant d’être testé dans 10 parties en mode visuel.

---

## 🔍 Analyse rapide

✅ **Points forts** :

* Simple, rapide à implémenter.
* Très efficace dans des environnements discrets comme Taxi-v3.

⚠️ **Limites** :

* Ne fonctionne pas bien si l’espace d’états ou d’actions est trop grand.
* Nécessite de stocker une Q-table complète (ici 500×6 = 3000 valeurs).

