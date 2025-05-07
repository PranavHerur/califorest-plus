import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss
import time
import random
from typing import List, Dict, Tuple, Any, Union, Optional
import copy
import os

from califorest.rfva import RFVA


class Particle:
    """
    Represents a particle in the MOPSO algorithm.
    Each particle corresponds to a set of Random Forest hyperparameters.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple],
        velocity_factor: float = 0.1,
        random_state: int = None,
    ):
        """
        Initialize a particle with random position and velocity.

        Args:
            bounds: Dictionary mapping parameter names to (min, max) bounds
            velocity_factor: Controls the initial velocity magnitude
            random_state: Random seed for reproducibility
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.bounds = bounds

        # Initialize position and velocity within bounds
        for param_name, (min_val, max_val) in bounds.items():
            if param_name == "max_features":
                # Special handling for max_features which can be float or string
                options = ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]
                self.position[param_name] = random.choice(options)
                self.velocity[param_name] = (
                    0  # For categorical variables, velocity is a probability
                )
            elif isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameters (n_estimators, max_depth, etc.)
                self.position[param_name] = np.random.randint(min_val, max_val + 1)
                max_velocity = velocity_factor * (max_val - min_val)
                self.velocity[param_name] = np.random.uniform(
                    -max_velocity, max_velocity
                )
            else:
                # Float parameters
                self.position[param_name] = np.random.uniform(min_val, max_val)
                max_velocity = velocity_factor * (max_val - min_val)
                self.velocity[param_name] = np.random.uniform(
                    -max_velocity, max_velocity
                )

        # Initialize personal best
        self.best_position = copy.deepcopy(self.position)

        # Initialize fitness values
        self.fitness = None
        self.best_fitness = None

    def update_velocity(
        self,
        global_best_position: Dict[str, Any],
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        """
        Update particle velocity based on inertia, cognitive, and social components.

        Args:
            global_best_position: The global best position (dictionary of hyperparameters)
            w: Inertia weight
            c1: Cognitive weight (personal best influence)
            c2: Social weight (global best influence)
        """
        for param_name in self.position:
            if param_name == "max_features":
                # For categorical variables, use a different approach
                if random.random() < 0.3:  # 30% chance to change
                    # With 50% probability, move toward global best
                    if random.random() < 0.5:
                        self.velocity[param_name] = global_best_position[param_name]
                    # Otherwise, explore a random option
                    else:
                        options = ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]
                        self.velocity[param_name] = random.choice(options)
            else:
                # For numerical parameters, use standard PSO velocity update
                r1, r2 = np.random.random(2)

                cognitive_velocity = (
                    c1
                    * r1
                    * (self.best_position[param_name] - self.position[param_name])
                )
                social_velocity = (
                    c2
                    * r2
                    * (global_best_position[param_name] - self.position[param_name])
                )

                self.velocity[param_name] = (
                    w * self.velocity[param_name] + cognitive_velocity + social_velocity
                )

    def update_position(self):
        """
        Update the particle's position based on velocity and bounds.
        """
        for param_name in self.position:
            if param_name == "max_features":
                # For categorical variables, directly use velocity as new value
                if (
                    self.velocity[param_name] != 0
                ):  # Only change if velocity indicates change
                    self.position[param_name] = self.velocity[param_name]
                    self.velocity[param_name] = 0  # Reset velocity
            else:
                # For numerical parameters
                min_val, max_val = self.bounds[param_name]

                # Update position
                new_position = self.position[param_name] + self.velocity[param_name]

                # Apply bounds and handle integer parameters
                if isinstance(min_val, int) and isinstance(max_val, int):
                    new_position = int(round(new_position))

                new_position = max(min_val, min(max_val, new_position))
                self.position[param_name] = new_position

    def is_dominated_by(self, other_fitness: List[float]) -> bool:
        """
        Check if this particle is dominated by another particle with the given fitness.
        A solution dominates another if it's at least as good in all objectives and better in at least one.

        Args:
            other_fitness: List of fitness values for another particle

        Returns:
            True if this particle is dominated by the other, False otherwise
        """
        if self.fitness is None:
            return True

        # Assuming first objective is to be maximized (e.g., AUC)
        # and other objectives are to be minimized (e.g., Brier score, training time)
        better_in_some = False

        for i, (self_val, other_val) in enumerate(zip(self.fitness, other_fitness)):
            if i == 0:  # AUC (maximize)
                if other_val > self_val:
                    better_in_some = True
                elif other_val < self_val:
                    return False
            else:  # Other metrics (minimize)
                if other_val < self_val:
                    better_in_some = True
                elif other_val > self_val:
                    return False

        return better_in_some


class Repository:
    """
    Repository to store non-dominated solutions (Pareto front).
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize an empty repository.

        Args:
            max_size: Maximum number of solutions to keep in the repository
        """
        self.solutions = []  # List of (position, fitness) tuples
        self.max_size = max_size

    def add_solution(self, position: Dict[str, Any], fitness: List[float]):
        """
        Add a solution to the repository if it's non-dominated.

        Args:
            position: Dictionary of hyperparameters
            fitness: List of fitness values
        """
        # Create a temporary particle to use the is_dominated_by method
        temp_particle = Particle({})
        temp_particle.fitness = fitness

        # Check if this solution is dominated by any in the repository
        dominated = False
        i = 0
        while i < len(self.solutions):
            repo_position, repo_fitness = self.solutions[i]

            # If the new solution dominates one in the repository, remove the dominated one
            if self._dominates(fitness, repo_fitness):
                self.solutions.pop(i)
                continue

            # If the new solution is dominated by one in the repository, don't add it
            if self._dominates(repo_fitness, fitness):
                dominated = True
                break

            i += 1

        # If not dominated, add to repository
        if not dominated:
            self.solutions.append((copy.deepcopy(position), fitness))

            # If repository is full, remove the solution in the most crowded region
            if len(self.solutions) > self.max_size:
                self._remove_crowded()

    def _dominates(self, fitness1: List[float], fitness2: List[float]) -> bool:
        """
        Check if fitness1 dominates fitness2.

        Args:
            fitness1: First list of fitness values
            fitness2: Second list of fitness values

        Returns:
            True if fitness1 dominates fitness2, False otherwise
        """
        better_in_some = False

        for i, (val1, val2) in enumerate(zip(fitness1, fitness2)):
            if i == 0:  # AUC (maximize)
                if val1 < val2:
                    return False
                if val1 > val2:
                    better_in_some = True
            else:  # Other metrics (minimize)
                if val1 > val2:
                    return False
                if val1 < val2:
                    better_in_some = True

        return better_in_some

    def _remove_crowded(self):
        """Remove a solution from the most crowded region using crowding distance."""
        if len(self.solutions) <= 1:
            return

        # Calculate crowding distance for each solution
        distances = self._calculate_crowding_distances()

        # Remove the solution with the smallest crowding distance
        min_idx = np.argmin(distances)
        self.solutions.pop(min_idx)

    def _calculate_crowding_distances(self) -> List[float]:
        """
        Calculate crowding distance for each solution in the repository.

        Returns:
            List of crowding distances
        """
        n_solutions = len(self.solutions)
        n_objectives = len(self.solutions[0][1])
        distances = [0.0] * n_solutions

        for obj_idx in range(n_objectives):
            # Extract fitness values for this objective
            values = [self.solutions[i][1][obj_idx] for i in range(n_solutions)]

            # Sort solutions by this objective
            sorted_indices = np.argsort(values)

            # Set boundary points to infinity
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            # Normalization factor to handle different scales
            obj_range = values[sorted_indices[-1]] - values[sorted_indices[0]]
            if obj_range == 0:
                obj_range = 1.0  # Avoid division by zero

            # Calculate distances for intermediate points
            for i in range(1, n_solutions - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]

                # Normalized distance between neighbors
                distance = (values[next_idx] - values[prev_idx]) / obj_range
                distances[idx] += distance

        return distances

    def select_leader(self, method: str = "random") -> Dict[str, Any]:
        """
        Select a leader from the repository.

        Args:
            method: Method to select leader ('random', 'roulette', 'crowding')

        Returns:
            Position of the selected leader
        """
        if not self.solutions:
            raise ValueError("Repository is empty, cannot select leader")

        if method == "random":
            # Randomly select a solution
            return copy.deepcopy(random.choice(self.solutions)[0])

        elif method == "roulette":
            # Select based on crowding distance (higher distance = higher probability)
            distances = self._calculate_crowding_distances()
            total = sum(distances)
            if total == 0:
                return copy.deepcopy(random.choice(self.solutions)[0])

            # Normalize distances to get probabilities
            probs = [d / total for d in distances]
            idx = np.random.choice(len(self.solutions), p=probs)
            return copy.deepcopy(self.solutions[idx][0])

        elif method == "crowding":
            # Select the solution in the least crowded region
            distances = self._calculate_crowding_distances()
            idx = np.argmax(distances)
            return copy.deepcopy(self.solutions[idx][0])

        else:
            raise ValueError(f"Unknown leader selection method: {method}")

    def get_pareto_front(self) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """
        Get the current Pareto front.

        Returns:
            Tuple of (positions, fitnesses)
        """
        positions = [sol[0] for sol in self.solutions]
        fitnesses = [sol[1] for sol in self.solutions]
        return positions, fitnesses


class MOPSO:
    """
    Multi-Objective Particle Swarm Optimization for Random Forest hyperparameter tuning.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple],
        n_particles: int = 20,
        max_iter: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        repository_size: int = 50,
        mutation_rate: float = 0.1,
        leader_selection_method: str = "crowding",
        random_state: Optional[int] = None,
    ):
        """
        Initialize MOPSO algorithm.

        Args:
            bounds: Dictionary mapping parameter names to (min, max) bounds
            n_particles: Number of particles in the swarm
            max_iter: Maximum number of iterations
            w: Inertia weight
            c1: Cognitive weight (personal best influence)
            c2: Social weight (global best influence)
            repository_size: Maximum number of solutions in the repository
            mutation_rate: Probability of mutation
            leader_selection_method: Method to select leader ('random', 'roulette', 'crowding')
            random_state: Random seed for reproducibility
        """
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.repository = Repository(max_size=repository_size)
        self.mutation_rate = mutation_rate
        self.leader_selection_method = leader_selection_method
        self.random_state = random_state
        self.history = []  # Track repository size over iterations

        # Initialize random state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        # Initialize particles
        self.particles = []
        for i in range(n_particles):
            self.particles.append(Particle(bounds, random_state=random_state))

    def evaluate_particle(
        self, particle: Particle, X_train, y_train, X_val=None, y_val=None, cv: int = 5
    ) -> List[float]:
        """
        Evaluate a particle on multiple objectives.

        Args:
            particle: Particle to evaluate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (if None, use cross-validation)
            y_val: Validation labels (if None, use cross-validation)
            cv: Number of cross-validation folds

        Returns:
            List of fitness values [auc, brier_score, training_time]
        """
        # Create Random Forest with the particle's hyperparameters
        params = copy.deepcopy(particle.position)
        rf = RFVA(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            random_state=self.random_state,
            n_jobs=-1,  # Use all available cores
        )

        # Measure training time
        start_time = time.time()
        rf.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Calculate AUC and Brier score
        if X_val is not None and y_val is not None:
            # Use validation set
            y_pred_proba = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            brier = brier_score_loss(y_val, y_pred_proba)
        else:
            # Use cross-validation
            cv_auc = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
            auc = np.mean(cv_auc)

            # For Brier score, we need to do manual cross-validation to get probabilities
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            y_probs = np.zeros_like(y_train, dtype=float)

            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                rf_cv = RFVA(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    max_features=params["max_features"],
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf_cv.fit(X_fold_train, y_fold_train)
                y_probs[val_idx] = rf_cv.predict_proba(X_fold_val)[:, 1]

            brier = brier_score_loss(y_train, y_probs)

        # Return fitness values [auc (maximize), brier_score (minimize), training_time (minimize)]
        return [auc, brier, training_time]

    def apply_mutation(self, particle: Particle):
        """
        Apply mutation to a particle with some probability.

        Args:
            particle: Particle to potentially mutate
        """
        if random.random() < self.mutation_rate:
            # Randomly select a parameter to mutate
            param_name = random.choice(list(particle.position.keys()))

            if param_name == "max_features":
                # For categorical variables, choose a new random option
                options = ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]
                particle.position[param_name] = random.choice(options)
            else:
                # For numerical parameters, add a random perturbation
                min_val, max_val = self.bounds[param_name]
                range_val = max_val - min_val

                # Perturb by up to 20% of the parameter range
                perturbation = random.uniform(-0.2, 0.2) * range_val
                new_val = particle.position[param_name] + perturbation

                # Apply bounds and handle integer parameters
                if isinstance(min_val, int) and isinstance(max_val, int):
                    new_val = int(round(new_val))

                particle.position[param_name] = max(min_val, min(max_val, new_val))

    def optimize(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        cv: int = 5,
        verbose: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """
        Run the MOPSO optimization algorithm.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            cv: Number of cross-validation folds
            verbose: Whether to print progress

        Returns:
            Tuple of (positions, fitnesses) representing the Pareto front
        """
        # Convert to numpy arrays if not already
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if X_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val)

        # Initialize repository with first generation
        for particle in self.particles:
            fitness = self.evaluate_particle(
                particle, X_train, y_train, X_val, y_val, cv
            )
            particle.fitness = fitness
            particle.best_fitness = fitness
            self.repository.add_solution(particle.position, fitness)

        self.history.append(len(self.repository.solutions))

        if verbose:
            print(f"Initial repository size: {len(self.repository.solutions)}")

        # Main loop
        for iteration in range(self.max_iter):
            if verbose:
                print(f"\nIteration {iteration + 1}/{self.max_iter}")

            # Update each particle
            for particle in self.particles:
                # Select a leader from the repository
                leader_position = self.repository.select_leader(
                    method=self.leader_selection_method
                )

                # Update velocity and position
                particle.update_velocity(leader_position, self.w, self.c1, self.c2)
                particle.update_position()

                # Apply mutation
                self.apply_mutation(particle)

                # Evaluate new position
                fitness = self.evaluate_particle(
                    particle, X_train, y_train, X_val, y_val, cv
                )
                particle.fitness = fitness

                # Update personal best if this position is non-dominated
                if not particle.is_dominated_by(particle.best_fitness):
                    particle.best_position = copy.deepcopy(particle.position)
                    particle.best_fitness = fitness

                # Add to repository
                self.repository.add_solution(particle.position, fitness)

            self.history.append(len(self.repository.solutions))

            if verbose:
                positions, fitnesses = self.repository.get_pareto_front()
                print(f"Repository size: {len(positions)}")
                if len(positions) > 0:
                    # Calculate average fitness values
                    avg_auc = np.mean([f[0] for f in fitnesses])
                    avg_brier = np.mean([f[1] for f in fitnesses])
                    avg_time = np.mean([f[2] for f in fitnesses])
                    print(
                        f"Average AUC: {avg_auc:.4f}, Brier: {avg_brier:.4f}, Time: {avg_time:.2f}s"
                    )

                    # Find best solution for each objective
                    best_auc_idx = np.argmax([f[0] for f in fitnesses])
                    best_brier_idx = np.argmin([f[1] for f in fitnesses])
                    best_time_idx = np.argmin([f[2] for f in fitnesses])

                    print(
                        f"Best AUC: {fitnesses[best_auc_idx][0]:.4f} - {positions[best_auc_idx]}"
                    )
                    print(
                        f"Best Brier: {fitnesses[best_brier_idx][1]:.4f} - {positions[best_brier_idx]}"
                    )
                    print(
                        f"Best Time: {fitnesses[best_time_idx][2]:.2f}s - {positions[best_time_idx]}"
                    )

        return self.repository.get_pareto_front()

    def plot_pareto_front(self, save_path: Optional[str] = None):
        """
        Plot the Pareto front for pairs of objectives.

        Args:
            save_path: Optional path to save the figure
        """
        positions, fitnesses = self.repository.get_pareto_front()

        if not fitnesses:
            print("No solutions in repository to plot")
            return

        # Extract fitness values for each objective
        aucs = [f[0] for f in fitnesses]
        briers = [f[1] for f in fitnesses]
        times = [f[2] for f in fitnesses]

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot AUC vs Brier Score
        axs[0].scatter(aucs, briers, c="blue", alpha=0.7)
        axs[0].set_xlabel("AUC (maximize)")
        axs[0].set_ylabel("Brier Score (minimize)")
        axs[0].set_title("AUC vs Brier Score")
        axs[0].grid(True, linestyle="--", alpha=0.7)

        # Plot AUC vs Training Time
        axs[1].scatter(aucs, times, c="green", alpha=0.7)
        axs[1].set_xlabel("AUC (maximize)")
        axs[1].set_ylabel("Training Time (minimize)")
        axs[1].set_title("AUC vs Training Time")
        axs[1].grid(True, linestyle="--", alpha=0.7)

        # Plot Brier Score vs Training Time
        axs[2].scatter(briers, times, c="red", alpha=0.7)
        axs[2].set_xlabel("Brier Score (minimize)")
        axs[2].set_ylabel("Training Time (minimize)")
        axs[2].set_title("Brier Score vs Training Time")
        axs[2].grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot the history of repository size over iterations.

        Args:
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.history)), self.history, "b-", marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Repository Size")
        plt.title("Repository Size vs Iteration")
        plt.grid(True, linestyle="--", alpha=0.7)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def get_best_solution(
        self, criterion: str = "auc"
    ) -> Tuple[Dict[str, Any], List[float]]:
        """
        Get the best solution from the Pareto front according to a specific criterion.

        Args:
            criterion: Criterion to select best solution ('auc', 'brier', 'time', 'balanced')

        Returns:
            Tuple of (position, fitness)
        """
        positions, fitnesses = self.repository.get_pareto_front()

        if not fitnesses:
            raise ValueError("No solutions in repository")

        if criterion == "auc":
            # Maximize AUC
            idx = np.argmax([f[0] for f in fitnesses])
        elif criterion == "brier":
            # Minimize Brier score
            idx = np.argmin([f[1] for f in fitnesses])
        elif criterion == "time":
            # Minimize training time
            idx = np.argmin([f[2] for f in fitnesses])
        elif criterion == "balanced":
            # Normalize and balance all objectives
            aucs = np.array([f[0] for f in fitnesses])
            briers = np.array([f[1] for f in fitnesses])
            times = np.array([f[2] for f in fitnesses])

            # Normalize to [0, 1] range
            auc_norm = (aucs - np.min(aucs)) / (np.max(aucs) - np.min(aucs) + 1e-10)
            brier_norm = 1 - (briers - np.min(briers)) / (
                np.max(briers) - np.min(briers) + 1e-10
            )
            time_norm = 1 - (times - np.min(times)) / (
                np.max(times) - np.min(times) + 1e-10
            )

            # Weighted sum (equal weights)
            balanced_score = 0.4 * auc_norm + 0.4 * brier_norm + 0.2 * time_norm
            idx = np.argmax(balanced_score)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return positions[idx], fitnesses[idx]

    def create_results_df(self) -> pd.DataFrame:
        """
        Create a DataFrame with all solutions in the repository.

        Returns:
            DataFrame with hyperparameters and fitness values
        """
        positions, fitnesses = self.repository.get_pareto_front()

        if not positions:
            return pd.DataFrame()

        # Create rows for DataFrame
        rows = []
        for pos, fit in zip(positions, fitnesses):
            row = copy.deepcopy(pos)
            row["auc"] = fit[0]
            row["brier_score"] = fit[1]
            row["training_time"] = fit[2]
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by AUC (descending)
        df = df.sort_values("auc", ascending=False)

        return df
