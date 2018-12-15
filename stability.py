import numpy as np
from posterior import *
from multiprocessing import Pool, cpu_count, Manager
import matplotlib.pyplot as plt
import json


def main(log, alpha, m, eps):
    np_file_name = "output/{:.3f}_{:02d}_{:.3f}.npy".format(alpha, m, eps)
    data = np.load(np_file_name)
    lambda_2 = abs(np.sort(np.linalg.eigvals(data))[-2])
    diags = np.diag(data)
    stability = np.mean(diags[:4]) / np.mean(diags[4:])
    log["{:.3f}_{:02d}_{:.3f}".format(alpha, m, eps)] = dict(lambda_2=lambda_2, stability=stability)


if __name__ == '__main__':
    worker_args = []
    log = Manager().dict()
    for alpha in [.5, .01]:
        for eps in [.05, .001]:
            for m in range(1, 11):
                worker_args.append((log, alpha, m, eps))

    with Pool(cpu_count()) as f:
        f.starmap(main, worker_args)

    with open("output/log.json", "w") as f:
        json.dump(dict(log), f, indent=4)

    plt.figure()
    with open("output/log.json") as f:
        data = json.load(f)
    for alpha in [.5, .01]:
        for eps in [.05, .001]:
            plt.plot(list(range(1, 11)),
                     list(data["{:.3f}_{:02d}_{:.3f}".format(alpha, m, eps)]["lambda_2"] for m in range(1, 11)),
                     label="alpha={:.3f}, eps={:.3f}".format(alpha, eps))
    plt.legend()
    plt.savefig("output/lambda_2.png")
    plt.clf()
    for alpha in [.5, .01]:
        for eps in [.05, .001]:
            plt.semilogy(list(range(1, 11)),
                         list(data["{:.3f}_{:02d}_{:.3f}".format(alpha, m, eps)]["stability"] for m in range(1, 11)),
                         label="alpha={:.3f}, eps={:.3f}".format(alpha, eps))
    plt.legend()
    plt.savefig("output/stability.png")
