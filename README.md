# ORSolver: Comprehensive Operations Research Solver

[![Tests](https://img.shields.io/badge/tests-61%20passed-success.svg)](file:///Users/baptistegrosjean/Source/UMons/ORSolver/tests)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.15-blue.svg)](pyproject.toml)
[![Methodology](https://img.shields.io/badge/methodology-100%25%20Pure%20TDD-brightgreen.svg)](tests)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**ORSolver** est une bibliothèque Python modulaire, robuste et **exhaustive** pour la **Recherche Opérationnelle (RO)**, l'**Aide à la Décision** et l'**Intelligence Artificielle Décisionnelle**.

Conçue initialement à l'**UMons (Université de Mons)**, cette bibliothèque a été intégralement développée selon la méthodologie **100% pure TDD (Test-Driven Development)** pour couvrir l'ensemble des branches académiques et industrielles de l'optimisation mathématique :
- **Programmation Linéaire** : Simplexe Primal 2-phases, Simplexe Dual, Dualité, Analyse de Sensibilité, Goal Programming
- **Programmation en Nombres Entiers** : Branch & Bound, Coupures de Gomory, Sac à dos 0-1
- **Transport & Affectation** : VAM, Moindre Coût, Nord-Ouest, Algorithme Hongrois
- **Optimisation de Réseaux** : Dijkstra, Bellman-Ford, Floyd-Warshall, Edmonds-Karp, Flot à Coût Min, MST (Kruskal/Prim), PERT/CPM, **A\***
- **Théorie des Jeux** : Jeux matriciels Minimax (von Neumann), **Arbres de jeu Minimax avec élagage $\alpha$-$\beta$**
- **Processus Décisionnels de Markov (MDP)** : Itération sur la valeur et sur les politiques (Bellman)
- **Aide à la Décision Multicritère (MCDA)** : **AHP (Saaty)** avec ratio de cohérence $CR$, **TOPSIS** avec solutions idéales
- **Théorie de la Décision dans l'Incertain & Risque** : Wald (Maximin), Maximax, Hurwicz, Savage (Minimax Regret), Laplace, EMV, EVPI
- **Fiabilité & Maintenance** : Systèmes Série, Parallèle, $k$-sur-$n$, Lois Exponentielle et Weibull, MTBF/MTTF
- **Programmation Dynamique (DP)** : Multiplication de chaîne de matrices, Rendu de monnaie optimal
- **Simulation de Risque** : Simulation de Monte Carlo pour projets PERT stochastiques
- **Théorie des Files d'Attente** : Modèles $M/M/1$ et $M/M/s$ (Formule d'Erlang-C)
- **Gestion des Stocks** : Formule de Wilson / EOQ, pénuries admises, point de commande et stock de sécurité
- **Métaheuristiques** : TSP par Plus Proche Voisin, 2-Opt et Recuit Simulé
- **Optimisation Non Linéaire** : Section Dorée 1D, Descente de Gradient avec recherche linéaire d'Armijo
- **Visualisation & Formats** : Méthode graphique 2D adaptative, Parser CPLEX `.lp` standard.

---

## 📑 Sommaire
- [Installation & Tests](#-installation--tests)
- [Guide des Modules & Exemples de Code](#-guide-des-modules--exemples-de-code)
  - [1. Programmation Linéaire (Simplexe 2-Phases)](#1-programmation-linéaire-simplexe-2-phases)
  - [2. Analyse de Sensibilité Post-Optimale](#2-analyse-de-sensibilité-post-optimale)
  - [3. Simplexe Dual](#3-simplexe-dual)
  - [4. Programmation Linéaire en Nombres Entiers (MIP)](#4-programmation-linéaire-en-nombres-entiers-mip)
  - [5. Transport & Affectation (Hongrois)](#5-transport--affectation-hongrois)
  - [6. Réseaux, Flots & Algorithme A\*](#6-réseaux-flots--algorithme-a)
  - [7. Théorie des Jeux (Matrice Minimax & Arbres Alpha-Beta)](#7-théorie-des-jeux-matrice-minimax--arbres-alpha-beta)
  - [8. Processus Décisionnels de Markov (MDP)](#8-processus-décisionnels-de-markov-mdp)
  - [9. Aide à la Décision Multicritère (AHP & TOPSIS)](#9-aide-à-la-décision-multicritère-ahp--topsis)
  - [10. Décision dans l'Incertain et le Risque (Wald, Savage, EVPI)](#10-décision-dans-lincertain-et-le-risque-wald-savage-evpi)
  - [11. Fiabilité des Systèmes (Série, Parallèle, k-sur-n, MTBF)](#11-fiabilité-des-systèmes-série-parallèle-k-sur-n-mtbf)
  - [12. Programmation Dynamique (Chaîne de matrices & Rendu de monnaie)](#12-programmation-dynamique-chaîne-de-matrices--rendu-de-monnaie)
  - [13. Simulation Monte Carlo PERT & Risque](#13-simulation-monte-carlo-pert--risque)
  - [14. Files d'Attente (M/M/1 et M/M/s)](#14-files-dattente-mm1-et-mms)
  - [15. Gestion des Stocks (Wilson / EOQ & ROP)](#15-gestion-des-stocks-wilson--eoq--rop)
  - [16. Métaheuristiques pour le TSP (2-Opt & Recuit Simulé)](#16-métaheuristiques-pour-le-tsp-2-opt--recuit-simulé)
  - [17. Optimisation Non Linéaire (NLP)](#17-optimisation-non-linéaire-nlp)
  - [18. Méthode Graphique 2D Dynamique](#18-méthode-graphique-2d-dynamique)
  - [19. Parser Standard CPLEX .lp](#19-parser-standard-cplex-lp)
- [Architecture du Dépôt](#-architecture-du-dépôt)
- [Rétro-compatibilité](#-rétro-compatibilité)

---

## 🚀 Installation & Tests

### Prérequis
- Python 3.9+
- Dépendances : `numpy`, `matplotlib`, `pytest`

```bash
git clone https://github.com/baptistegrosjean/ORSolver.git
cd ORSolver
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Exécuter la suite complète de tests (100% Pure TDD)
```bash
pytest -v
```
**61 tests unitaires** couvrant l'ensemble des modules, algorithmes et cas limites.

---

## 💡 Guide des Modules & Exemples de Code

### 9. Aide à la Décision Multicritère (AHP & TOPSIS)
```python
from orsolver.mcda import solve_ahp, solve_topsis

# 1. Méthode AHP de Saaty (matrice de comparaisons par paires)
pairwise = [
    [1.0, 3.0, 5.0],
    [1.0 / 3.0, 1.0, 2.0],
    [1.0 / 5.0, 0.5, 1.0],
]
ahp = solve_ahp(pairwise)
print("Poids de priorité AHP :", ahp.weights)
print(f"Ratio de cohérence CR = {ahp.consistency_ratio:.4f} (Cohérent: {ahp.is_consistent})")

# 2. Méthode TOPSIS (classement multi-critères d'alternatives)
matrix = [
    [250.0, 16.0, 12.0, 5.0],
    [200.0, 16.0, 8.0, 3.0],
    [300.0, 32.0, 16.0, 4.0],
]
weights = [0.25, 0.25, 0.25, 0.25]
types = ["cost", "benefit", "benefit", "benefit"]
topsis = solve_topsis(matrix, weights, types)
print("Scores TOPSIS :", topsis.scores)
print("Classement (meilleur au pire) :", topsis.ranking)
```

---

### 10. Décision dans l'Incertain et le Risque (Wald, Savage, EVPI)
```python
from orsolver.decision import decision_under_uncertainty, decision_under_risk

payoffs = [
    [10.0, 20.0, 30.0],
    [5.0, 40.0, 15.0],
    [15.0, 15.0, 15.0],
]

# Sans probabilités : Wald, Maximax, Hurwicz, Savage Minimax Regret
unc = decision_under_uncertainty(payoffs, hurwicz_alpha=0.6)
print(f"Meilleure action selon Maximin (Wald) : A{unc.maximin_action + 1}")
print(f"Meilleure action selon Savage Regret : A{unc.savage_action + 1} (Regret max = {unc.savage_min_max_regret})")

# Avec probabilités : EMV et Valeur de l'Information Parfaite (EVPI)
risk = decision_under_risk(payoffs, probabilities=[0.3, 0.5, 0.2])
print(f"Action optimale (EMV) : A{risk.best_action + 1} avec gain espéré = {risk.best_emv}")
print(f"Valeur de l'information parfaite (EVPI) = {risk.evpi:.2f}")
```

---

### 11. Fiabilité des Systèmes (Série, Parallèle, k-sur-n, MTBF)
```python
from orsolver.reliability import (
    series_reliability,
    parallel_reliability,
    k_out_of_n_reliability,
    exponential_reliability,
)

# Système Série & Parallèle
r = [0.95, 0.90, 0.85]
print("Fiabilité Série :", series_reliability(r))
print("Fiabilité Parallèle :", parallel_reliability(r))

# Redondance 2-sur-3
print("Fiabilité 2-sur-3 :", k_out_of_n_reliability(k=2, n=3, component_reliability=0.9))

# Loi exponentielle
r_t, mttf = exponential_reliability(rate_lambda=0.001, t=100.0)
print(f"Fiabilité à t=100h : {r_t:.4f}, MTTF = {mttf:.1f} heures")
```

---

### 12. Programmation Dynamique (Chaîne de matrices & Rendu de monnaie)
```python
from orsolver.dynamic_programming import matrix_chain_multiplication, coin_change

# 1. Parenthésage optimal de matrices
dims = [10, 30, 5, 60]  # A1: 10x30, A2: 30x5, A3: 5x60
min_mult, parens = matrix_chain_multiplication(dims)
print(f"Nombre minimal de multiplications scalaires : {min_mult}, Ordre : {parens}")

# 2. Rendu de monnaie optimal
nb_coins, breakdown = coin_change(coins=[1, 2, 5, 10, 20, 50], amount=67)
print(f"Monnaie pour 67 : {nb_coins} pièces utilisées -> {breakdown}")
```

---

### 13. Simulation Monte Carlo PERT & Risque
```python
from orsolver.simulation import monte_carlo_pert, StochasticTask

tasks = [
    StochasticTask(name="A", optimistic=2.0, most_likely=3.0, pessimistic=4.0),
    StochasticTask(name="B", optimistic=3.0, most_likely=4.0, pessimistic=5.0),
    StochasticTask(name="C", optimistic=1.0, most_likely=2.0, pessimistic=3.0, predecessors=["A"]),
    StochasticTask(name="D", optimistic=4.0, most_likely=5.0, pessimistic=6.0, predecessors=["B"]),
]

sim_res = monte_carlo_pert(tasks, num_simulations=2000, seed=42)
print(f"Durée moyenne du projet : {sim_res.mean_duration:.2f} jours (Écart-type: {sim_res.std_dev:.2f})")
print("Indices de criticité des tâches :", sim_res.criticality_index)
print(f"Probabilité de finir en <= 10 jours : {sim_res.probability_of_completion(10.0):.1%}")
```

---

## 🏛 Architecture du Dépôt

```
ORSolver/
├── orsolver/
│   ├── core/               # Abstractions (Problem, Variable, Constraint, Solution)
│   ├── lp/                 # Simplexe 2-phases, Simplexe Dual, Dualité, Sensibilité, Goal Programming
│   ├── mip/                # Branch and Bound (MILP), Coupures de Gomory, Sac à dos 0-1
│   ├── transport/          # Transport (VAM, Moindre Coût, Nord-Ouest) & Affectation (Hongrois)
│   ├── network/            # Dijkstra, Bellman-Ford, Floyd, Flot Max, Flot Coût Min, MST, PERT/CPM, A*
│   ├── game_theory/        # Jeux matriciels Minimax & Arbres de jeu Alpha-Beta
│   ├── mdp/                # Processus de Markov (Itération Valeur & Politique)
│   ├── mcda/               # Aide à la décision multicritère (AHP de Saaty & TOPSIS)
│   ├── decision/           # Théorie de la décision (Wald, Maximax, Hurwicz, Savage, Laplace, EVPI)
│   ├── reliability/        # Fiabilité industrielle (Série, Parallèle, k-sur-n, Exponentielle, Weibull)
│   ├── dynamic_programming/# Programmation dynamique (Multiplication de chaîne de matrices, Rendu de monnaie)
│   ├── simulation/         # Simulation de Monte Carlo pour l'analyse de risque PERT
│   ├── queuing/            # Théorie des files d'attente (M/M/1 et M/M/s Erlang-C)
│   ├── inventory/          # Gestion des stocks (Modèle de Wilson, pénuries, point de commande)
│   ├── heuristics/         # TSP: Plus proche voisin, 2-Opt, Recuit Simulé
│   ├── nlp/                # Optimisation non-linéaire: Section dorée 1D, Descente de gradient
│   ├── visualizer/         # Méthode graphique 2D auto-scalée
│   └── cli/                # Parser standard CPLEX .lp
├── tests/                  # Suite de tests unitaires 100% pure TDD (61 tests)
├── GraphicalORSolver.py    # Wrapper rétro-compatible
├── SimplexSolver.py        # Wrapper rétro-compatible
├── ORSolver.py             # Script principal historique
├── pyproject.toml          # Fichier de packaging standard Python
└── requirements.txt        # Dépendances du projet
```
