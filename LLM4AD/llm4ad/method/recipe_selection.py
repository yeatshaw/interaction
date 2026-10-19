from __future__ import annotations

import math
import random
import threading
from collections import deque

import numpy as np


class RecipeSelector:
    """Shared multiple-play selector for adaptive reflection recipes."""

    STRATEGIES = {
        "linear_thompson_sampling",
        "sliding_window_thompson_sampling",
        "discounted_mab",
        "sliding_window_mab",
    }
    STATE_DIM = 5

    def __init__(self, recipe_ids, strategy="linear_thompson_sampling", *,
                 prior_precision=1.0, posterior_scale=0.5,
                 discount_factor=0.9, window_size=50, exploration=1.0,
                 context_dim=5):
        self.recipe_ids = tuple(str(value) for value in recipe_ids)
        self.strategy = str(strategy).lower()
        if self.strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown recipe selection strategy: {strategy}")
        self.prior_precision = float(prior_precision)
        self.posterior_scale = float(posterior_scale)
        self.discount_factor = float(discount_factor)
        self.window_size = int(window_size)
        self.exploration = float(exploration)
        self.context_dim = int(context_dim)
        if self.context_dim < 1:
            raise ValueError("context_dim must be positive")
        if self.prior_precision <= 0 or self.posterior_scale <= 0:
            raise ValueError("Thompson sampling parameters must be positive")
        if not 0.0 < self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be in (0, 1]")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.exploration < 0:
            raise ValueError("exploration must be non-negative")
        self._lock = threading.RLock()
        self._reset()

    def _reset(self):
        dim = self.context_dim + 1
        self.A = {key: self.prior_precision * np.eye(dim)
                  for key in self.recipe_ids}
        self.b = {key: np.zeros(dim, dtype=float) for key in self.recipe_ids}
        self.trials = {key: 0 for key in self.recipe_ids}
        self.discounted_count = {key: 0.0 for key in self.recipe_ids}
        self.discounted_reward = {key: 0.0 for key in self.recipe_ids}
        self.window_rewards = {key: deque(maxlen=self.window_size)
                               for key in self.recipe_ids}
        # Sliding-window Thompson keeps raw (context, reward) pairs so the
        # posterior is rebuilt from recent evidence only. Once the window is
        # full the oldest sample is dropped, which lets the sampled
        # distribution follow non-stationary recipe payoffs.
        self.window_thompson_features = {
            key: deque(maxlen=self.window_size) for key in self.recipe_ids}
        self.window_thompson_rewards = {
            key: deque(maxlen=self.window_size) for key in self.recipe_ids}
        self.recent_arms = deque(maxlen=self.window_size)
        self.total_updates = 0

    def _features(self, state):
        values = np.asarray(list(state), dtype=float)
        if values.shape != (self.context_dim,) or not np.all(np.isfinite(values)):
            raise ValueError(f"context must contain {self.context_dim} finite values")
        return np.concatenate(([1.0], values))

    def _sample_linear_value(self, recipe_id, features):
        inverse = np.linalg.pinv(self.A[recipe_id])
        mean = inverse @ self.b[recipe_id]
        covariance = self.posterior_scale ** 2 * inverse
        covariance = (covariance + covariance.T) / 2.0
        covariance += 1e-10 * np.eye(covariance.shape[0])
        theta = np.random.multivariate_normal(mean, covariance, check_valid="ignore")
        return float(features @ theta)

    def _window_posterior(self, recipe_id):
        dim = self.context_dim + 1
        matrix = self.prior_precision * np.eye(dim)
        vector = np.zeros(dim, dtype=float)
        for features, reward in zip(self.window_thompson_features[recipe_id],
                                    self.window_thompson_rewards[recipe_id]):
            matrix += np.outer(features, features)
            vector += float(reward) * features
        return matrix, vector

    def _sample_window_value(self, recipe_id, features):
        matrix, vector = self._window_posterior(recipe_id)
        inverse = np.linalg.pinv(matrix)
        mean = inverse @ vector
        covariance = self.posterior_scale ** 2 * inverse
        covariance = (covariance + covariance.T) / 2.0
        covariance += 1e-10 * np.eye(covariance.shape[0])
        theta = np.random.multivariate_normal(mean, covariance, check_valid="ignore")
        return float(features @ theta)

    def _mab_values(self, candidates):
        values = {}
        if self.strategy == "discounted_mab":
            unseen = [arm for arm in candidates
                      if self.discounted_count[arm] <= 1e-12]
            total = sum(self.discounted_count.values())
            for arm in candidates:
                count = self.discounted_count[arm]
                mean = (self.discounted_reward[arm] / count
                        if count > 1e-12 else 0.0)
                bonus = (self.exploration * math.sqrt(
                    math.log1p(total) / count) if count > 1e-12 else float("inf"))
                values[arm] = (float("inf") if arm in unseen
                               else mean + bonus)
            return values

        seen = set(self.recent_arms)
        total = len(self.recent_arms)
        unseen = [arm for arm in candidates if arm not in seen]
        for arm in candidates:
            rewards = self.window_rewards[arm]
            count = len(rewards)
            mean = sum(rewards) / count if count else 0.0
            bonus = (self.exploration * math.sqrt(
                math.log1p(total) / count) if count else float("inf"))
            values[arm] = (float("inf") if arm in unseen
                           else mean + bonus)
        return values

    def select_top_k(self, state, candidate_ids, k=1):
        candidates = list(dict.fromkeys(str(value) for value in candidate_ids))
        if not candidates:
            return [], {}
        unknown = set(candidates).difference(self.recipe_ids)
        if unknown:
            raise KeyError(f"Unknown recipe ids: {sorted(unknown)}")
        k = min(max(int(k), 0), len(candidates))
        if not k:
            return [], {}
        with self._lock:
            if self.strategy == "linear_thompson_sampling":
                features = self._features(state)
                values = {arm: self._sample_linear_value(arm, features)
                          for arm in candidates}
                unseen = [arm for arm in candidates if self.trials[arm] == 0]
                selected = random.sample(unseen, min(k, len(unseen)))
                selected_set = set(selected)
                remaining = sorted(
                    (arm for arm in candidates if arm not in selected_set),
                    key=values.__getitem__, reverse=True)
                selected.extend(remaining[:k - len(selected)])
                return selected, values

            if self.strategy == "sliding_window_thompson_sampling":
                features = self._features(state)
                values = {arm: self._sample_window_value(arm, features)
                          for arm in candidates}
                unseen = [arm for arm in candidates if self.trials[arm] == 0]
                selected = random.sample(unseen, min(k, len(unseen)))
                selected_set = set(selected)
                remaining = sorted(
                    (arm for arm in candidates if arm not in selected_set),
                    key=values.__getitem__, reverse=True)
                selected.extend(remaining[:k - len(selected)])
                return selected, values

            values = self._mab_values(candidates)
            random_ties = {arm: random.random() for arm in candidates}
            selected = sorted(
                candidates, key=lambda arm: (values[arm], random_ties[arm]),
                reverse=True)[:k]
            return selected, values

    def select_population_top_k(self, states, candidate_ids, k=3):
        """One independent Thompson sample and one vote per individual."""
        if self.strategy != "linear_thompson_sampling":
            raise ValueError("Individual voting currently requires linear_thompson_sampling")
        candidates = list(dict.fromkeys(str(value) for value in candidate_ids))
        unknown = set(candidates).difference(self.recipe_ids)
        if unknown:
            raise KeyError(f"Unknown recipe ids: {sorted(unknown)}")
        if not candidates:
            return [], {"vote_counts": {}, "individual_votes": []}
        if len(states) == 0:
            raise ValueError("Individual voting requires a non-empty population")
        features = np.asarray([self._features(state) for state in states])
        k = min(max(int(k), 0), len(candidates))
        with self._lock:
            scores = []
            for arm in candidates:
                inverse = np.linalg.pinv(self.A[arm])
                mean = inverse @ self.b[arm]
                covariance = self.posterior_scale ** 2 * inverse
                covariance = (covariance + covariance.T) / 2.0
                covariance += 1e-10 * np.eye(covariance.shape[0])
                theta = np.random.multivariate_normal(
                    mean, covariance, size=len(features), check_valid="ignore")
                scores.append(np.sum(theta * features, axis=1))
            scores = np.stack(scores, axis=-1)  # [individual, recipe]
            # Randomize exact score ties rather than favoring recipe order.
            ties = scores == np.max(scores, axis=-1, keepdims=True)
            winners = np.argmax(np.where(ties, np.random.random(scores.shape), -1), axis=-1)
            votes = {arm: int(np.count_nonzero(winners == index))
                     for index, arm in enumerate(candidates)}
            tie_breaks = {arm: random.random() for arm in candidates}
            selected = sorted(
                candidates, key=lambda arm: (votes[arm], tie_breaks[arm]),
                reverse=True)[:k]
        return selected, {
            "vote_counts": votes,
            "individual_votes": [candidates[index] for index in winners],
            "samples_per_individual": 1,
        }

    def update_population(self, recipe_id, states, reward):
        """Use parent individuals with total observation weight one per edge.

        The population reward is a shared label, not an individually measured
        effect. Fractional weights avoid counting it as N independent trials.
        """
        if self.strategy != "linear_thompson_sampling":
            raise ValueError("Individual voting currently requires linear_thompson_sampling")
        if recipe_id not in self.trials:
            raise KeyError(f"Unknown recipe_id: {recipe_id}")
        reward = float(reward)
        if not math.isfinite(reward) or len(states) == 0:
            raise ValueError("Population update needs a finite reward and non-empty states")
        features = np.asarray([self._features(state) for state in states])
        with self._lock:
            self.A[recipe_id] += features.T @ features / len(features)
            self.b[recipe_id] += reward * np.mean(features, axis=0)
            self.trials[recipe_id] += 1
            self.total_updates += 1

    def update(self, recipe_id, state, reward):
        recipe_id = str(recipe_id)
        if recipe_id not in self.trials:
            raise KeyError(f"Unknown recipe_id: {recipe_id}")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("recipe reward must be finite")
        with self._lock:
            if self.strategy == "linear_thompson_sampling":
                features = self._features(state)
                self.A[recipe_id] += np.outer(features, features)
                self.b[recipe_id] += reward * features
            elif self.strategy == "sliding_window_thompson_sampling":
                features = self._features(state)
                self.window_thompson_features[recipe_id].append(features)
                self.window_thompson_rewards[recipe_id].append(reward)
            elif self.strategy == "discounted_mab":
                decay = self.discount_factor
                for arm in self.recipe_ids:
                    self.discounted_count[arm] *= decay
                    self.discounted_reward[arm] *= decay
                self.discounted_count[recipe_id] += 1.0
                self.discounted_reward[recipe_id] += reward
            else:
                self.window_rewards[recipe_id].append(reward)
                self.recent_arms.append(recipe_id)
            self.trials[recipe_id] += 1
            self.total_updates += 1

    def state_dict(self):
        with self._lock:
            return {
                "strategy": self.strategy,
                "context_dim": self.context_dim,
                "recipe_ids": list(self.recipe_ids),
                "prior_precision": self.prior_precision,
                "posterior_scale": self.posterior_scale,
                "discount_factor": self.discount_factor,
                "window_size": self.window_size,
                "exploration": self.exploration,
                "A": {key: value.tolist() for key, value in self.A.items()},
                "b": {key: value.tolist() for key, value in self.b.items()},
                "trials": dict(self.trials),
                "discounted_count": dict(self.discounted_count),
                "discounted_reward": dict(self.discounted_reward),
                "window_rewards": {
                    key: list(value) for key, value in self.window_rewards.items()},
                "window_thompson_features": {
                    key: [[float(item) for item in vector]
                          for vector in self.window_thompson_features[key]]
                    for key in self.recipe_ids},
                "window_thompson_rewards": {
                    key: [float(value) for value in self.window_thompson_rewards[key]]
                    for key in self.recipe_ids},
                "recent_arms": list(self.recent_arms),
                "total_updates": self.total_updates,
            }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        if int(state.get("context_dim", 5)) != self.context_dim:
            raise ValueError("selector context dimension does not match checkpoint")
        saved_ids = tuple(str(value) for value in state.get("recipe_ids", ()))
        if saved_ids and saved_ids != self.recipe_ids:
            raise ValueError("selector recipe ids do not match checkpoint")
        saved_strategy = state.get("strategy")
        if saved_strategy and saved_strategy != self.strategy:
            raise ValueError("selector strategy does not match checkpoint")
        with self._lock:
            for key in self.recipe_ids:
                for name, target in (("A", self.A), ("b", self.b)):
                    if key in state.get(name, {}):
                        target[key] = np.asarray(state[name][key], dtype=float)
                for name, target in (("trials", self.trials),
                                     ("discounted_count", self.discounted_count),
                                     ("discounted_reward", self.discounted_reward)):
                    if key in state.get(name, {}):
                        target[key] = type(target[key])(state[name][key])
                if key in state.get("window_rewards", {}):
                    self.window_rewards[key] = deque(
                        state["window_rewards"][key], maxlen=self.window_size)
                if key in state.get("window_thompson_features", {}):
                    self.window_thompson_features[key] = deque(
                        (np.asarray(vector, dtype=float)
                         for vector in state["window_thompson_features"][key]),
                        maxlen=self.window_size)
                if key in state.get("window_thompson_rewards", {}):
                    self.window_thompson_rewards[key] = deque(
                        (float(value)
                         for value in state["window_thompson_rewards"][key]),
                        maxlen=self.window_size)
            self.recent_arms = deque(
                state.get("recent_arms", ()), maxlen=self.window_size)
            self.total_updates = int(state.get("total_updates", sum(self.trials.values())))


def create_recipe_selector(recipe_ids, config=None, **overrides):
    options = dict(config or {})
    options.update(overrides)
    allowed = {
        "strategy", "prior_precision", "posterior_scale", "discount_factor",
        "window_size", "exploration", "context_dim",
    }
    options = {key: value for key, value in options.items()
               if key in allowed}
    return RecipeSelector(recipe_ids, **options)
