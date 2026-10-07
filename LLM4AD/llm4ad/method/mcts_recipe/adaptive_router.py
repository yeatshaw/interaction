from __future__ import annotations

import random

import numpy as np


class ContextualThompsonRouter:
    """Multiple-play contextual Thompson sampler for recipe selection.

    Each recipe owns a linear-Gaussian reward model. The router samples one
    parameter vector per candidate recipe and returns the top-k distinct
    recipes. Context features are intentionally kept raw to match the
    experiment definition.
    """

    STATE_DIM = 5

    def __init__(self, recipe_ids, prior_precision=1.0,
                 posterior_scale=0.5):
        self.recipe_ids = tuple(str(x) for x in recipe_ids)
        self.prior_precision = float(prior_precision)
        self.posterior_scale = float(posterior_scale)
        if self.prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        if self.posterior_scale <= 0:
            raise ValueError("posterior_scale must be positive")
        self._reset()

    def _reset(self):
        dimension = self.STATE_DIM + 1  # five raw features plus intercept
        self.A = {
            recipe_id: self.prior_precision * np.eye(dimension)
            for recipe_id in self.recipe_ids
        }
        self.b = {
            recipe_id: np.zeros(dimension, dtype=float)
            for recipe_id in self.recipe_ids
        }
        self.trials = {recipe_id: 0 for recipe_id in self.recipe_ids}

    @staticmethod
    def _features(state):
        values = np.asarray(list(state), dtype=float)
        if values.shape != (ContextualThompsonRouter.STATE_DIM,):
            raise ValueError(
                f"context state must contain exactly "
                f"{ContextualThompsonRouter.STATE_DIM} values")
        if not np.all(np.isfinite(values)):
            raise ValueError("context state must contain finite values")
        return np.concatenate(([1.0], values))

    def _sample_value(self, recipe_id, features):
        A = self.A[recipe_id]
        b = self.b[recipe_id]
        A_inv = np.linalg.pinv(A)
        theta_hat = A_inv @ b
        covariance = self.posterior_scale ** 2 * A_inv
        covariance = (covariance + covariance.T) / 2.0
        covariance += 1e-10 * np.eye(covariance.shape[0])
        theta_sample = np.random.multivariate_normal(
            theta_hat, covariance, check_valid="ignore")
        return float(features @ theta_sample)

    def select_top_k(self, state, candidate_ids, k=3):
        """Select k distinct recipes using multiple-play Thompson sampling."""
        candidates = [str(x) for x in candidate_ids]
        if not candidates:
            return [], {}
        k = min(int(k), len(candidates))
        features = self._features(state)

        # Guarantee cold-start coverage before relying on posterior samples.
        untried = [recipe_id for recipe_id in candidates
                   if self.trials.get(recipe_id, 0) == 0]
        selected = random.sample(untried, min(k, len(untried)))

        sampled_values = {}
        for recipe_id in candidates:
            if recipe_id in selected:
                sampled_values[recipe_id] = float("inf")
            else:
                sampled_values[recipe_id] = self._sample_value(
                    recipe_id, features)

        remaining = [recipe_id for recipe_id in candidates
                     if recipe_id not in selected]
        remaining.sort(key=lambda recipe_id: sampled_values[recipe_id],
                       reverse=True)
        selected.extend(remaining[:k - len(selected)])
        return selected, sampled_values

    def update(self, recipe_id, state, reward):
        recipe_id = str(recipe_id)
        if recipe_id not in self.A:
            raise KeyError(f"Unknown recipe_id: {recipe_id}")
        features = self._features(state)
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("router reward must be finite")
        self.A[recipe_id] += np.outer(features, features)
        self.b[recipe_id] += reward * features
        self.trials[recipe_id] += 1

    def state_dict(self):
        return {
            "recipe_ids": list(self.recipe_ids),
            "prior_precision": self.prior_precision,
            "posterior_scale": self.posterior_scale,
            "state_dim": self.STATE_DIM,
            "A": {key: value.tolist() for key, value in self.A.items()},
            "b": {key: value.tolist() for key, value in self.b.items()},
            "trials": dict(self.trials),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        saved_ids = tuple(str(x) for x in state.get("recipe_ids", ()))
        if saved_ids and saved_ids != self.recipe_ids:
            raise ValueError(
                "adaptive router recipe ids do not match the checkpoint")
        saved_A = state.get("A", {})
        saved_b = state.get("b", {})
        saved_trials = state.get("trials", {})
        for recipe_id in self.recipe_ids:
            if recipe_id in saved_A:
                self.A[recipe_id] = np.asarray(saved_A[recipe_id], dtype=float)
            if recipe_id in saved_b:
                self.b[recipe_id] = np.asarray(saved_b[recipe_id], dtype=float)
            if recipe_id in saved_trials:
                self.trials[recipe_id] = int(saved_trials[recipe_id])
