# Network PPO — Guide technique

[Retour à la présentation](../README.md)

Un repo **PPO uniquement** pour décider du **placement des services, du routage des demandes et des quantités de trafic servies** sur un réseau industriel simplifié. Il n'importe ni `mecbench`, ni ses autres solveurs (MILP, A2C, CMA-ES, etc.).

## Démarrage

Prérequis : **Python 3.10+**. Depuis la racine du repo :

```bash
python -m pip install -e .
python train.py
python eval.py
```

Les **10 fichiers** `instances/train/*.json` constituent le jeu d'entraînement. L'entraînement sauvegarde :

```text
outputs/ppo_factory/model.zip
outputs/ppo_factory/vecnormalize.pkl
outputs/ppo_factory/metadata.json
```

L'évaluation charge ce checkpoint et lit par défaut `instances/eval/factory_shift_unseen.json`, scénario **non utilisé** lors de l'entraînement. Elle produit **`outputs/decision.json`**, avec la solution faisable et ses contraintes contrôlées.

Pour un test rapide de la tuyauterie (pas un apprentissage significatif) :

```bash
python train.py --timesteps 1024 --seed 42
python eval.py --instance instances/eval/factory_shift_unseen.json
```

Paramétrage explicite :

```bash
python train.py --instances instances/train --output outputs/experiment_1 \
  --timesteps 100000 --episode-steps 96 --paths 3 --seed 42

python eval.py --checkpoint outputs/experiment_1 \
  --instance /path/to/industrial_instance.json \
  --output outputs/my_decision.json --episodes 5 --stochastic --seed 123
```

Les sorties sont des **fichiers locaux**, pas des changements appliqués automatiquement à un orchestrateur.

## Structure

```text
network-ppo/
├── train.py                     # entraîne PPO (10 instances)
├── eval.py                      # charge PPO et prédit sur 1 instance JSON
├── src/network_ppo/
│   ├── problem.py               # domaine : nœuds, liens, services, demandes
│   ├── instance_io.py           # import JSON + contrôles de format
│   ├── paths.py                 # chemins candidats déterministes
│   ├── constraints.py           # vérification des contraintes réseau
│   ├── heuristic.py             # état initial faisable + référence simple
│   ├── environment.py           # Gymnasium (actions PPO)
│   └── solution_io.py           # décision validée, export JSON
├── instances/train/             # exactement 10 scénarios synthétiques
├── instances/eval/              # scénario synthétique tenu à l'écart
├── tests/
├── pyproject.toml
└── Dockerfile
```

## Fournir une instance industrielle

Il suffit de créer un **JSON** conforme au format v1. Exemple minimal (2 nœuds, 1 demande) :

```json
{
  "schema_version": 1,
  "id": "industrial_example",
  "units": {"flow": "Mbps", "latency": "ms", "compute": "vCPU"},
  "nodes": [
    {"id": "gateway", "cpu": 0},
    {"id": "edge", "cpu": 8}
  ],
  "links": [
    {"source": "gateway", "target": "edge", "capacity_mbps": 20,
     "latency_base_ms": 2, "latency_per_mbps_ms": 0.1}
  ],
  "services": [
    {"id": "vision", "cpu": 4, "demands": [
      {"id": "camera_1", "origin": "gateway", "min_mbps": 1,
       "max_mbps": 6, "max_latency_ms": 10}
    ]}
  ]
}
```

**Attention : ce petit exemple montre seulement le schéma JSON. Il ne peut PAS être évalué avec le modèle `ppo_factory`**, car il n'a pas la même topologie que le jeu d'entraînement. Pour exécuter `eval.py` avec le checkpoint fourni par `train.py`, il faut reprendre la liste **ordonnée** des nœuds, liens, services et demandes du dossier `instances/train/`, puis modifier les **valeurs** : CPU disponibles, capacités, latences, origines, min/max Mbps, budgets SLA, etc. Les identifiants, l'ordre et l'association demande → service doivent rester identiques. Un réseau complètement nouveau nécessite un nouvel entraînement avec 10 scénarios compatibles (ou une V2 à architecture graphe / padding).

Voir `instances/eval/factory_shift_unseen.json` pour un exemple complet de scénario de production **illustratif** pour l'usine de démonstration (passerelles nord/sud, edge, switch, cloud).

### Conventions métier

- Liens physiques **bidirectionnels**. Chaque direction a une capacité indépendante, égale à `capacity_mbps`.
- Latence d'un arc = `latency_base_ms + latency_per_mbps_ms × charge_totale_de_l'arc_mbps`. La latence d'une demande est la somme sur sa route.
- Un service est placé **une seule fois** sur un nœud. Sa consommation CPU n'est comptée qu'une fois, quel que soit le nombre de demandes.
- Chaque demande choisit un des **K plus courts chemins simples en nombre de sauts** ; s'il existe moins de K chemins, le dernier est réutilisé. Aucun partage multipath.
- Chaque demande reçoit un nombre **entier de Mbps**, entre `min_mbps` et `max_mbps`, inclus.
- Objectif : maximiser la **somme des Mbps servis** tout en respectant les CPU, capacités des arcs et budgets de latence. Pas de coût de migration ni de pondération des services.

### Actions et observation PPO

L'espace d'actions est **discret**, sans champs inutilisés : déplacer un service sur un nœud, choisir un chemin pour une demande, ou modifier sa quantité de **±1 Mbps**. Tout mouvement infaisable est **refusé**, laissant la solution précédente inchangée. La récompense correspond à la variation de Mbps servis (petite pénalité pour une action inutile/invalide).

L'observation inclut placements, routes, débits, latences, charges CPU et liens, capacités et budgets, origines et mapping service/demande. Les observations sont normalisées pendant l'entraînement via `VecNormalize`, puis avec les **mêmes statistiques** pendant l'évaluation.

Chaque épisode démarre d'un **état faisable à débit minimum**, construit par une recherche bornée (`heuristic.py`). Cela garantit qu'une décision exportée passe le validateur, mais implique que la politique RL apprend une **amélioration à partir d'une initialisation** et ne construit pas seule une solution depuis zéro. Une recherche bornée peut échouer même si une solution existe ; le programme remonte alors une erreur explicite.

## Format de sortie

`outputs/decision.json` inclut notamment :

- `feasible`, `total_served_mbps`, `max_requested_mbps` et `served_fraction` ;
- `placements`: service → nœud + vCPU utilisés ;
- `demands`: service, origine, route (nœuds dans l'ordre), Mbps alloués, latence calculée et budget ;
- `node_compute_loads`: vCPU utilisés et disponibles ;
- `directed_links`: trafic et capacité par **arc dirigé** ;
- `reference_seed_greedy_mbps`: quantité servie par une amélioration gloutonne des quantités avec les mêmes placement/routes initiaux, **sans PPO** ;
- `ppo_minus_reference_mbps`: écart de PPO par rapport à cette référence (qui peut être **négatif**).

Un seul épisode avec `--stochastic` absent produit des actions **déterministes**. Avec `--episodes 5 --stochastic`, le meilleur état valide rencontré est retenu sur plusieurs épisodes.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Les tests couvrent chargement JSON, compatibilité des 10 scénarios, recherche d'un seed faisable, violations CPU/latence/débit, direction des liens, export, ainsi que les actions Gymnasium lorsque cette dépendance est installée.

## Docker (CPU)

```bash
docker build -t network-ppo .
docker run --rm -v "$(pwd)/outputs:/app/outputs" network-ppo python train.py --timesteps 100000
docker run --rm -v "$(pwd)/outputs:/app/outputs" network-ppo python eval.py
```

Pour un JSON local en dehors du repo, monter son répertoire (`-v /chemin/instances:/app/external:ro`) et passer `--instance /app/external/fichier.json`. Les poids sont conservés entre deux conteneurs grâce au volume `outputs/`.

## Limitations / statut

- Les 10 scénarios et l'usine sont **synthétiques** : ce ne sont pas des données opérateur. PPO sur 10 cas ne prouve pas une généralisation industrielle.
- Topologie, ordre de nœuds, liens, services et demandes **fixes** entre entraînement et évaluation : une nouvelle architecture de réseau impose un réentraînement (V1).
- Pas de multipath, files d'attente réalistes, jitter, haute disponibilité, réplication, priorités de service, coût de migration, énergie ou déploiement live.
- PPO est **une heuristique apprise**, pas une garantie d'optimalité. Comparer ses résultats sur plusieurs seeds aux références déterministes et à un solveur exact **dans un benchmark séparé**.
- Les figures et comparaisons exploratoires sont disponibles dans le [notebook](../notebook/network_and_efficiency.ipynb). Un court entraînement vérifie le pipeline, mais ne suffit pas à mesurer la qualité d'une politique.

Le repo ne dépend d'aucun fichier, sous-module ni paquet Python du benchmark d'origine.
