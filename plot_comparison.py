import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    plt.figure()
    with open("output/log.json") as f:
        data = json.load(f)
    for alpha in [.5, .01]:
        for eps in [.05, .001]:
            plt.plot(list(range(1, 11)),
                     list(data["{:.3f}_{:02d}_{:.3f}".format(alpha, m, eps)]["lambda_2"] for m in range(1, 11)))
    plt.savefig("output/lambda_2.png")
    plt.clf()
    for alpha in [.5, .01]:
        for eps in [.05, .001]:
            plt.semilogy(list(range(1, 11)),
                     list(data["{:.3f}_{:02d}_{:.3f}".format(alpha, m, eps)]["stability"] for m in range(1, 11)))
    plt.savefig("output/stability.png")
