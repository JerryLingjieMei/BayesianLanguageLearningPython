import numpy as np
from utils import *


def ys_on_hsxs(hypotheses, xs, ys, eps):
    """
    :param hypotheses: h x 4 x 4
    :param xs: b x m x 4
    :param ys: b x m x 4
    :param eps:
    :return: b x h
    """
    # b x m x h x 4
    pred = xs.dot(hypotheses)
    # b x m x h
    eq = np.sum(pred & np.expand_dims(ys, 2), 3)
    # b x m x h
    weight = eq * (1 - eps) + (1 - eq) * eps / 3
    # b x h
    return np.prod(weight, 1)


def hs_new_on_xsys(prior, ys_on_hsxs_val):
    """
    :param prior: h
    :param ys_on_hsxs_val: b x h
    :return: b x h
    """
    # b x h
    post = ys_on_hsxs_val * prior
    return post / np.sum(post, 1, keepdims=True)


def hs_new_on_hs_approx(hypotheses, prior, m, eps):
    xs = get_random_sequence(m, N_SAMPLES)
    ys = get_random_sequence(m, N_SAMPLES)
    # b x h
    ys_on_hsxs_val = ys_on_hsxs(hypotheses, xs, ys, eps)
    # b x h
    hs_new_on_xsys_val = hs_new_on_xsys(prior, ys_on_hsxs_val)
    trans = np.inner(np.transpose(ys_on_hsxs_val), np.transpose(hs_new_on_xsys_val)) / 1000
    normalized_trans = trans / np.sum(trans, 1, keepdims=True)
    return normalized_trans


def hs_new_on_hs_non_approx(hypotheses, prior, m, eps):
    crude_xs = make_cartesian_product(PATTERNS_MATRIX, m)
    crude_ys = make_cartesian_product(PATTERNS_MATRIX, m)
    xs = np.tile(crude_xs, (len(crude_ys), 1, 1))
    ys = np.repeat(crude_xs, len(crude_ys), 0)
    # b x h
    ys_on_hsxs_val = ys_on_hsxs(hypotheses, xs, ys, eps)
    # b x h
    hs_new_on_xsys_val = hs_new_on_xsys(prior, ys_on_hsxs_val)
    trans = np.inner(np.transpose(ys_on_hsxs_val), np.transpose(hs_new_on_xsys_val)) / crude_xs.shape[0]
    return trans


def hs_new_on_hs(hypotheses, prior, m, eps):
    if m <= 3:
        return hs_new_on_hs_non_approx(hypotheses, prior, m, eps)
    else:
        return hs_new_on_hs_approx(hypotheses, prior, m, eps)
