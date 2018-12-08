import numpy as np
from posterior import *
from multiprocessing import Pool, cpu_count, Manager
import json


def main(log, alpha, m, eps):
    np_file_name = "output/{:.3f}_{:02d}_{:.3f}.npy".format(alpha, m, eps)
    data = np.load(np_file_name)
    lambda_2 = abs(np.sort(np.linalg.eigvals(data))[-2])
    norm = np.linalg.norm(data)
    log["{:.3f}_{:02d}_{:.3f}".format(alpha, m, eps)] = dict(lambda_2=lambda_2, stability=norm)


if __name__ == '__main__':
    worker_args = []
    log = Manager().dict()
    for alpha in [.5, .01]:
        for eps in [.05, .01]:
            for m in range(1, 11):
                worker_args.append((log, alpha, m, eps))

    with Pool(cpu_count()) as f:
        f.starmap(main, worker_args)

    with open("output/log.json", "w") as f:
        json.dump(dict(log), f, indent=4)
