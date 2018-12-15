import numpy as np
from posterior import *
from multiprocessing import Pool, cpu_count, Manager
import matplotlib.pyplot as plt
import json
from mcmc import *
from prior import get_prior


def main(alpha, m, eps):
    np_file_name = "output/{:.3f}_{:02d}_{:.3f}.npy".format(alpha, m, eps)
    data = np.load(np_file_name)
    plt.figure()
    plt.matshow(data[:8, :8], cmap="YlGn")
    plt.savefig("output/{:.3f}_{:02d}_{:.3f}_selected.png".format(alpha, m, eps))
    histories = metropolis_hastings(data)
    plt.figure()
    plt.scatter(np.arange(0, N_ITERATIONS, 10), histories[0::10], s=1)
    plt.savefig("output/{:.3f}_{:02d}_{:.3f}_chain.png".format(alpha, m, eps))
    plt.figure()
    plt.hist(histories, bins=260, density=1, label="chain")
    plt.plot(np.arange(0, 260), get_prior(alpha), label="prior")
    plt.legend()
    plt.savefig("output/{:.3f}_{:02d}_{:.3f}_hist.png".format(alpha, m, eps))


if __name__ == '__main__':
    worker_args = [(.5, 1, .05), (.5, 3, .05), (.5, 10, .05), (.01, 3, .05)]

    with Pool(cpu_count()) as f:
        f.starmap(main, worker_args)
