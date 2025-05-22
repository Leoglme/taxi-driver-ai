# 🎲 Monte Carlo Agent pour Taxi-v3 🚕

Ce module contient une implémentation d’un **agent Monte Carlo** pour résoudre l’environnement `Taxi-v3` de Gymnasium.

## 📚 Description de l’algorithme

L’algorithme Monte Carlo est une méthode **model-free** d’apprentissage par renforcement, qui se base sur l’expérience **complète d’un épisode** (du début à la fin) pour mettre à jour les estimations des valeurs d’action (**Q-values**).

Monte Carlo apprend à la fin de l’épisode.
Il regarde tout ce qu’il s’est passé et calcule la récompense totale reçue pour chaque action prise.

Contrairement au Q-Learning (qui met à jour à chaque étape), Monte Carlo :
- **observe la totalité de l’épisode**,
- **calcule le retour total `G`** pour chaque (état, action),
- puis **met à jour la Q-table** uniquement à la première visite de chaque paire (état, action).

---

## 🔁 Mise à jour Monte Carlo (First-Visit)

À la fin d’un épisode, pour chaque `(état, action)` rencontré pour la première fois dans l’épisode :

```

G = rₜ + γ \* rₜ₊₁ + γ² \* rₜ₊₂ + ...
Q(s, a) ← Moyenne de tous les retours G observés pour (s, a)

````

Cela permet une **estimation stable**, mais nécessite d’attendre la fin de chaque épisode pour apprendre.

---

## ⚙️ Paramètres principaux

| Paramètre         | Valeur par défaut | Description                                      |
|-------------------|-------------------|--------------------------------------------------|
| `discount_factor` | `0.99`            | Pondération des récompenses futures              |
| `epsilon`         | `1.0`             | Taux d’exploration initial                       |
| `epsilon_decay`   | `0.999`           | Réduction progressive de l’exploration           |
| `min_epsilon`     | `0.01`            | Valeur minimale d’epsilon                        |
| `num_episodes`    | `50000`           | Nombre d’épisodes d’apprentissage                |
| `max_steps`       | `200`             | Nombre max d’actions par épisode                 |

---

## 📂 Composants du code

- `generate_episode(env, max_steps)` : génère une trajectoire complète de l’agent (états, actions, récompenses).
- `update(episode)` : met à jour les Q-values à partir du retour total `G`, en appliquant la méthode **First-Visit Monte Carlo**.
- `choose_action(state)` : applique une politique **epsilon-greedy** pour équilibrer exploration et exploitation.

---

## 🧠 Avantages et limites

✅ **Forces** :
- Simple à implémenter.
- Fournit des estimations stables sur le long terme.
- Ne nécessite pas de connaissance du modèle de l’environnement.

⚠️ **Limites** :
- Nécessite **la fin de l’épisode** pour apprendre.
- Moins efficace dans des environnements avec **longs épisodes ou grande variance**.
- Moins rapide que le Q-learning pour converger dans certains cas.

---

## ✅ Objectif du projet

Ce Monte Carlo agent est destiné à être comparé à d’autres approches, comme :

- **Q-Learning** (déjà implémenté)
- **Deep Q-Learning** (à venir)
- **Brute Force** (baseline naïve)

L’objectif est d’évaluer leurs performances (vitesse de convergence, nombre d’étapes, récompenses moyennes...) sur un même environnement.

---

## 📈 Exemple d’affichage

```bash
[MonteCarlo] Épisode 3000, ε=0.421
````

---

## 🧪 Utilisation

Depuis `main.py` :

```bash
$ python main.py
=== Choix de l'algorithme ===
1) Q-learning
2) Monte Carlo
Entrez 1 ou 2 puis [Entrée] : 2
```

L’agent s’entraîne puis est testé visuellement sur 10 épisodes consécutifs.
