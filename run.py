from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from prior import *
from posterior import *


def main(alpha, m, eps):
    hypotheses = get_hypotheses()
    prior = get_prior(alpha)
    result = hs_new_on_hs(hypotheses, prior, m, eps)
    file_name = "output/{:.3f}_{:02d}_{:.3f}.npy".format(alpha, m, eps)
    np.save(file_name, result)
    plt.clf()
    plt.figure()
    np_file_name = "output/{:.3f}_{:02d}_{:.3f}.npy".format(alpha, m, eps)
    data = np.load(np_file_name)
    plt.matshow(data.T)
    file_name = "output/{:.3f}_{:02d}_{:.3f}.png".format(alpha, m, eps)
    plt.savefig(file_name)
    print(file_name, " generated")


if __name__ == '__main__':
    worker_args = []
    for alpha in [.5, .01]:
        for eps in [.05, .001]:
            for m in range(1, 11):
                worker_args.append((alpha, m, eps))

    with Pool(cpu_count()) as f:
        f.starmap(main, worker_args)
