import itertools
import numpy as np

from utils import *


def get_compositional():
    result = np.zeros((4 ** 4, 4), dtype=np.uint8)
    for i, prod in enumerate(itertools.product(*([PATTERNS] * 4))):
        result[i] = prod
    return result


def get_holistic():
    result = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        result[i] = [(x // 2 + i // 2) % 2 * 2 + (x % 2 + i % 2) % 2 for x in range(4)]
    return result


def get_hypotheses():
    hypotheses_indices = np.concatenate((get_holistic(), get_compositional()), axis=0)
    result = np.zeros((hypotheses_indices.shape[0], hypotheses_indices.shape[1], 4), dtype=np.bool)
    for i, hypotheses_index in enumerate(hypotheses_indices):
        result[i] = PATTERNS_MATRIX[hypotheses_index]
    return result


def get_prior(alpha):
    return np.concatenate((np.full((4,), alpha / 4), np.full((256,), (1 - alpha) / 256)))
