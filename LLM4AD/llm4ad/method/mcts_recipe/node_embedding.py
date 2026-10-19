"""Generation-response features for population-node recipe selection."""

import numpy as np


class NodeEmbeddingError(RuntimeError):
    """An embedding failure must not be treated as an invalid LLM program."""


class NodeEmbedding:
    def __init__(self, encoder, dimensions):
        self.encoder = encoder
        self.dimensions = int(dimensions)
        if self.dimensions < 1:
            raise ValueError("node embedding dimensions must be positive")

    def validate(self, vector):
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.dimensions,) or not np.all(np.isfinite(value)):
            raise ValueError(
                f"Expected a finite node embedding with {self.dimensions} dimensions")
        return value

    def encode_response(self, response):
        """Embed the complete evolution response immediately after generation."""
        try:
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Generation response must be non-empty text")
            return self.validate(self.encoder.encode(response)).tolist()
        except Exception as exc:
            status = getattr(exc, "code", None)
            reason = f"HTTP {status}" if status is not None else type(exc).__name__
            raise NodeEmbeddingError(f"Failed to embed evolution response ({reason})") from exc

    def encode_population(self, population):
        missing = []
        for function in population.individuals:
            cached = getattr(function, "_node_embedding", None)
            if cached is not None:
                self.validate(cached)
                continue
            response = getattr(function, "_generation_raw_response", None)
            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    "Node embedding requires the original generation response; "
                    "start a new experiment instead of resuming a legacy node")
            missing.append(function)
        if missing:
            vectors = np.asarray(self.encoder.encode([
                function._generation_raw_response for function in missing
            ]), dtype=float)
            if vectors.shape != (len(missing), self.dimensions):
                raise ValueError(
                    "Embedding service returned an unexpected shape: "
                    f"{vectors.shape}, expected {(len(missing), self.dimensions)}")
            for function, vector in zip(missing, vectors):
                function._node_embedding = self.validate(vector).tolist()
        if not population.individuals:
            raise ValueError("Cannot embed an empty population")
        # A recipe acts on the population, so pool its surviving algorithms.
        return np.mean([
            self.validate(function._node_embedding)
            for function in population.individuals
        ], axis=0).tolist()
