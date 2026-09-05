import numpy as np


class GetData:
    def __init__(self, n_instance, n_cities, seed=2024):
        self.n_instance = int(n_instance)
        self.n_cities = int(n_cities)
        self.seed = int(seed)
        if self.n_instance < 1:
            raise ValueError("n_instance must be positive.")
        if self.n_cities < 2:
            raise ValueError("n_cities must be at least 2.")

    def generate_instances(self):
        # A local generator keeps task construction reproducible without
        # modifying NumPy's process-wide random state used by MCTS.
        rng = np.random.default_rng(self.seed)
        instance_data = []
        for _ in range(self.n_instance):
            coordinates = rng.random((self.n_cities, 2))
            distances = np.linalg.norm(
                coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :],
                axis=2,
            )
            instance_data.append((coordinates, distances))
        return instance_data
