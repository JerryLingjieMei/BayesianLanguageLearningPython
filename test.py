from utils import *
from prior import *
from posterior import *
import run
import unittest


class TestBasics(unittest.TestCase):
    def setUp(self):
        self.alpha = .5

    def test_utils(self):
        make_cartesian_product(PATTERNS_MATRIX, 4)
        get_random_sequence(3, 100)

    def test_hypotheses(self):
        get_hypotheses()

    def test_prior(self):
        get_prior(self.alpha)


class TestPosterior(unittest.TestCase):
    def setUp(self):
        self.hypotheses = get_hypotheses()
        self.prior = get_prior(.5)
        self.m = 1
        xs = np.zeros((100, self.m, 4), dtype=bool)
        xs[:, :, 0] = 1
        self.xs = xs
        ys = np.zeros((100, self.m, 4), dtype=bool)
        ys[:, :, 0] = 1
        self.ys = ys
        self.eps = .3

    def test_ys_on_hsxs(self):
        ys_on_hsxs(self.hypotheses, self.xs, self.ys, self.eps)

    def test_hs_new_on_xsys(self):
        hs_new_on_xsys(self.prior, ys_on_hsxs(self.hypotheses, self.xs, self.ys, self.eps))

    def test_hs_new_on_hs_approx(self):
        hs_new_on_hs_approx(self.hypotheses, self.prior, self.m, self.eps)

    def test_full_small(self):
        run.main(.01, 3, .05)

    def test_full_large(self):
        run.main(.5, 4, .1)


if __name__ == '__main__':
    unittest.main(verbosity=3)
