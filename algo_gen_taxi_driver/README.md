# 🚖 Taxi Driver AI - Optimisé

Ce projet implémente un agent intelligent pour l’environnement **Taxi-v3** de `gymnasium`, en combinant un **réseau neuronal GRU** et un **algorithme génétique**. L’objectif est d’entraîner un taxi à chercher un passager et le déposer au bon endroit de manière efficace.

Une interface graphique basique (Tkinter) permet de visualiser l’apprentissage et de tester l’agent optimisé.

---

## 🧠 Fonctionnalités

- Réseau neuronal GRU (mémoire de court terme)
- Algorithme génétique :
  - Sélection des meilleurs agents
  - Croisement (moyenne pondérée des poids)
  - Mutation des paramètres
- Récompenses ajustées pour guider efficacement l’apprentissage
- Interface Tkinter pour suivre les générations
- Test visuel de l’agent entraîné

---

## 📦 Installation

### ✅ Dépendances requises

```bash
pip install torch gymnasium numpy
```

Tkinter est inclus avec Python sur la plupart des systèmes. Sinon :

- **Debian/Ubuntu** : `sudo apt-get install python3-tk`
- **macOS** : `brew install python-tk`
- **Windows** : déjà intégré

---

## 🚀 Exécution

Lancez le script principal :

```bash
python taxi_driver_ai.py
```

- Une fenêtre s’ouvrira avec les logs de l’apprentissage génération par génération.
- Une fois l’entraînement terminé, un test sera automatiquement lancé avec affichage visuel.

---

## 🧩 Structure du projet

| Composant         | Description                                                     |
|-------------------|-----------------------------------------------------------------|
| `RNNNet`          | Réseau neuronal avec une couche GRU et une sortie linéaire      |
| `Individual`      | Agent évolutif avec son propre réseau et score de fitness       |
| `evaluate()`      | Fonction d’évaluation basée sur les performances en jeu         |
| `genetic_algorithm()` | Algorithme d’évolution avec sélection, croisement, mutation |
| `TaxiApp`         | Interface Tkinter pour lancer et visualiser l'entraînement      |
| `test_agent()`    | Phase de test finale dans un environnement rendu                |

---

## ⚙️ Logique d’apprentissage

- **Normalisation des entrées** du réseau pour assurer la stabilité :
  ```python
  [taxi_row / 4.0, taxi_col / 4.0, passenger_pos / 4.0, destination / 3.0]
  ```
- **Distance de Manhattan** utilisée pour juger la proximité de la cible
- **Récompenses personnalisées** :
  - +30 pour un bon ramassage
  - +3 ou -5 selon l’approche vers la cible
  - -50 pour inactivité / -100 pour boucle

---

## 📈 Exemple de sortie

```text
Génération 0 | Best: 32.5 | Moyenne: 10.2
Génération 1 | Best: 68.0 | Moyenne: 34.7
...
✅ Entraînement terminé.
Episode terminé en 89 étapes.
```

---

## 💡 Idées d’amélioration

- Sauvegarde / chargement du meilleur modèle
- Graphique matplotlib pour suivre la progression
- Paramètres configurables via ligne de commande
- Ajout d’un système d’élitisme ou d’adaptation dynamique du taux de mutation

---

## 👨‍💻 Auteur

Développé avec ❤️ en Python, PyTorch, Gymnasium et Tkinter.

---

> Ce projet est un excellent point de départ pour explorer l’apprentissage par renforcement hybride, combinant réseaux de neurones et optimisation évolutive.
