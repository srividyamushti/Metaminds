"""
Multi-Objective Evolutionary Optimization Engine
Aligns with IEEE Computational Intelligence Society (CIS) Evolutionary Computing principles.
Optimizes hyperparameters to find Pareto-optimal configurations balancing accuracy, latency, and complexity.
"""
from typing import Dict, List, Tuple
import numpy as np

# Safe import fallback for both root and src directory structures
try:
    from src.model import SensorFaultClassifier
except ImportError:
    from model import SensorFaultClassifier


class EvolutionaryOptimizer:
    def __init__(self, population_size: int = 8, generations: int = 5):
        self.population_size = population_size
        self.generations = generations

    def compute_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Scalarized multi-objective fitness formulation:
        Rewards high F1 score, penalizes excessive tree complexity and execution latency.
        """
        f1 = metrics["f1_score"]
        complexity_penalty = metrics["model_complexity"] * 0.003
        latency_penalty = metrics["latency_ms"] * 0.02
        return float(f1 - complexity_penalty - latency_penalty)

    def optimize(
        self, X_train, y_train, X_val, y_val
    ) -> Tuple[List[int], Dict[str, float], List[Dict]]:
        """
        Runs genetic search across generations: Selection -> Crossover -> Mutation.
        Chromosome: [max_depth (2-14), min_samples_split (2-12)]
        """
        np.random.seed(42)
        population = [
            [int(np.random.randint(2, 12)), int(np.random.randint(2, 10))]
            for _ in range(self.population_size)
        ]

        history: List[Dict] = []
        best_chromosome: List[int] = population[0]
        best_metrics: Dict[str, float] = {}
        best_fitness_val: float = -float("inf")

        for gen in range(self.generations):
            evaluated_population = []

            for chromo in population:
                clf = SensorFaultClassifier(max_depth=chromo[0], min_samples_split=chromo)
                clf.train(X_train, y_train)
                res = clf.evaluate(X_val, y_val)
                fitness = self.compute_fitness(res)
                evaluated_population.append((chromo, fitness, res))

                if fitness > best_fitness_val:
                    best_fitness_val = fitness
                    best_chromosome = chromo
                    best_metrics = res

            # Sort descending by fitness (Elitism)
            evaluated_population.sort(key=lambda x: x, reverse=True)
            elites = [item[0] for item in evaluated_population[: max(2, self.population_size // 2)]]

            # Next generation reproduction and mutation
            next_generation = list(elites)
            while len(next_generation) < self.population_size:
                parent = elites[np.random.randint(0, len(elites))]
                mutated = [
                    int(np.clip(parent[0] + np.random.choice(), 2, 14)),
                    int(np.clip(parent + np.random.choice(), 2, 12)),
                ]
                next_generation.append(mutated)

            population = next_generation
            history.append({
                "generation": gen + 1,
                "best_fitness": best_fitness_val,
                "best_f1": best_metrics.get("f1_score", 0.0),
                "best_latency_ms": best_metrics.get("latency_ms", 0.0),
            })

        return best_chromosome, best_metrics, history