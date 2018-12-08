import itertools
import numpy as np

PATTERNS = [0, 1, 2, 3]

PATTERNS_MATRIX = np.identity(4, dtype=np.bool)

N_SAMPLES = 260 * 1000


def make_cartesian_product(patterns_matrices, m):
    result = np.zeros((len(patterns_matrices) ** m, m, 4), dtype=np.uint8)
    for i, prod in enumerate(itertools.product(*([patterns_matrices] * m))):
        result[i, :, :] = prod
    return result


def get_random_sequence(m, batch_size):
    result = np.zeros((batch_size * m, 4), dtype=np.uint8)
    result[np.arange(batch_size * m), np.random.choice(4, batch_size * m)] = 1
    return result.reshape((batch_size, m, 4))
