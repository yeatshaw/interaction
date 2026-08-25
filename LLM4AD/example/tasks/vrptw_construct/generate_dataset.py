from pathlib import Path
import pickle
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    output = ROOT.parent / 'dataset' / 'vrptw_instances.pkl'
    output.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(2024)
    data = []
    for _ in range(100):
        n = 50
        coordinates = np.random.rand(n + 1, 2)
        demands = np.append([0], np.random.randint(1, 10, size=n))
        capacity = 40
        distances = np.linalg.norm(coordinates[:, None] - coordinates, axis=2)
        service = np.append([0], np.random.rand(n) * 0.05 + 0.15)
        length = np.random.rand(n) * 0.05 + 0.15
        d0i = distances[0, 1:]
        early_factor = np.random.rand(n) * (((4.6 - service[1:] - length) / d0i - 1) - 1) + 1
        early = early_factor * d0i
        windows = np.vstack(([0, 4.6], np.column_stack((early, early + length))))
        data.append((coordinates.tolist(), distances.tolist(), demands.tolist(),
                     int(capacity), service.tolist(), windows.tolist()))
    with output.open('wb') as file:
        pickle.dump(data, file)
    print(f'Wrote {len(data)} instances to {output}')
